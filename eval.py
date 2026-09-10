import os

os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.solver.gd import GradientSolver
import torch
from torchvision.transforms import v2 as transforms


# ---------------------------------------------------------------------------
# PATCH: stable-worldmodel 0.1.1 bug — GradientSolver.init_action() /
# prepare_init_action() leaves warm-start action tensor on CPU on the
# first planning call when init_action is None.
#
# This patch ensures the action tensor is created directly with the solver
# dtype/device before it is registered as the optimization parameter.
def _patched_init_action(self, n_envs, actions=None):
    if actions is None:
        actions = torch.zeros(
            (n_envs, 0, self.action_dim),
            dtype=self.dtype,
        )

    actions = actions.to(self.device)

    # Add the sample dimension to whatever warm-started actions we already have:
    # [n_envs, t, action_dim] -> [n_envs, num_samples, t, action_dim]
    actions = actions.unsqueeze(1).repeat_interleave(self.num_samples, dim=1)

    remaining = self.horizon - actions.shape[2]

    if remaining > 0:
        # Match the training-time noise prior from lejepa_forward:
        #   eps = torch.randn_like(ctx_actions_raw)          # eps ~ N(0, I)
        #   act_gamma = gamma * ctx_actions_raw + (1 - gamma) * eps
        # At gamma=0 the model only ever sees pure eps, so N(0, I) is the
        # correct "no information yet" init — not zeros. Sampling directly
        # at (n_envs, num_samples, ...) also gives each of the num_samples
        # candidates its own independent draw, instead of one draw repeated.
        new_actions = torch.randn(
            n_envs,
            self.num_samples,
            remaining,
            self.action_dim,
            dtype=self.dtype,
            device=self.device,
        )

        actions = torch.cat([actions, new_actions], dim=2)

    # Reuse existing parameter storage when possible.
    if hasattr(self, "init") and self.init.shape == actions.shape:
        self.init.copy_(actions)
    else:
        if "init" in self._parameters:
            del self._parameters["init"]

        self.register_parameter(
            "init",
            torch.nn.Parameter(actions)
        )


GradientSolver.init_action = _patched_init_action
# ---------------------------------------------------------------------------


def img_transform(cfg):
    """
    Transform images into the representation expected by the world model.
    """
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                **spt.data.dataset_stats.ImageNet
            ),
            transforms.Resize(
                size=cfg.eval.img_size
            ),
        ]
    )


def get_episodes_length(dataset, episodes):
    """
    Return the length of each requested episode.
    """

    col_name = (
        "episode_idx"
        if "episode_idx" in dataset.column_names
        else "ep_idx"
    )

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")

    lengths = [
        np.max(step_idx[episode_idx == ep_id]) + 1
        for ep_id in episodes
    ]

    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    """
    Load the HDF5 dataset and cache the requested columns.
    """

    dataset_path = Path(
        cfg.get("cache_dir")
        or swm.data.utils.get_cache_dir()
    )

    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


@hydra.main(
    version_base=None,
    config_path="./config/eval",
    config_name="pusht",
)
def run(cfg: DictConfig):
    """
    Run evaluation of the learned world model policy.
    """

    # -----------------------------------------------------------------------
    # Validate planning configuration
    # -----------------------------------------------------------------------
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), (
        "Planning horizon must be smaller than or equal to eval_budget"
    )

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Create world environment
    # -----------------------------------------------------------------------
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget

    world = swm.World(
        **cfg.world,
        image_shape=(224, 224),
    )

    # -----------------------------------------------------------------------
    # Image transforms
    # -----------------------------------------------------------------------
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    dataset = get_dataset(
        cfg,
        cfg.eval.dataset_name,
    )

    stats_dataset = dataset

    col_name = (
        "episode_idx"
        if "episode_idx" in dataset.column_names
        else "ep_idx"
    )

    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name),
        return_index=True,
    )

    # -----------------------------------------------------------------------
    # Fit preprocessing scalers
    # -----------------------------------------------------------------------
    process = {}

    for col in cfg.dataset.keys_to_cache:

        if col == "pixels":
            continue

        processor = preprocessing.StandardScaler()

        col_data = stats_dataset.get_col_data(col)

        # Remove rows containing NaNs before fitting the scaler.
        col_data = col_data[
            ~np.isnan(col_data).any(axis=1)
        ]

        processor.fit(col_data)

        process[col] = processor

        # Goal statistics use the same scaler as the corresponding state.
        if col != "action":
            process[f"goal_{col}"] = processor

    # -----------------------------------------------------------------------
    # Determine policy
    # -----------------------------------------------------------------------
    policy_name = cfg.get(
        "policy",
        "random",
    )

    if policy_name != "random":

        ckpt_path = policy_name

        if not ckpt_path.endswith(".ckpt"):
            ckpt_path += ".ckpt"

        print(
            f"Loading local PyTorch model from {ckpt_path}..."
        )

        # -------------------------------------------------------------------
        # Load checkpoint on CPU first.
        # -------------------------------------------------------------------
        model = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        # -------------------------------------------------------------------
        # Move model to device.
        # -------------------------------------------------------------------
        model = model.to(device)

        # -------------------------------------------------------------------
        # Evaluation mode.
        # -------------------------------------------------------------------
        model.eval()

        # -------------------------------------------------------------------
        # The world-model parameters should not be updated during planning.
        # The action variable itself is optimized by GradientSolver.
        # -------------------------------------------------------------------
        model.requires_grad_(False)

        # -------------------------------------------------------------------
        # ViT positional interpolation.
        # -------------------------------------------------------------------
        model.interpolate_pos_encoding = True

        # -------------------------------------------------------------------
        # Construct planning configuration.
        # -------------------------------------------------------------------
        config = swm.PlanConfig(
            **cfg.plan_config
        )

        # -------------------------------------------------------------------
        # Construct gradient-based planner.
        # -------------------------------------------------------------------
        solver = hydra.utils.instantiate(
            cfg.solver,
            model=model,
            device=device,
        )

        # -------------------------------------------------------------------
        # Construct policy wrapper.
        # -------------------------------------------------------------------
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
        )

    else:

        policy = swm.policy.RandomPolicy()

    # -----------------------------------------------------------------------
    # Determine result directory
    # -----------------------------------------------------------------------
    results_path = (
        Path(
            swm.data.utils.get_cache_dir(),
            cfg.policy,
        ).parent
        if policy_name != "random"
        else Path(__file__).resolve().parent
    )

    # -----------------------------------------------------------------------
    # Determine valid evaluation starting points
    # -----------------------------------------------------------------------
    episode_len = get_episodes_length(
        dataset,
        ep_indices,
    )

    max_start_idx = (
        episode_len
        - cfg.eval.goal_offset_steps
        - 1
    )

    max_start_idx_dict = {
        ep_id: max_start_idx[i]
        for i, ep_id in enumerate(ep_indices)
    }

    all_row_ep_indices = dataset.get_col_data(
        col_name
    )

    max_start_per_row = np.array(
        [
            max_start_idx_dict[ep_id]
            for ep_id in all_row_ep_indices
        ]
    )

    valid_mask = (
        dataset.get_col_data("step_idx")
        <= max_start_per_row
    )

    valid_indices = np.nonzero(
        valid_mask
    )[0]

    print(
        f"{len(valid_indices)} valid starting points "
        f"found for evaluation."
    )

    # -----------------------------------------------------------------------
    # Check that we have enough evaluation samples.
    # -----------------------------------------------------------------------
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Requested {cfg.eval.num_eval} evaluations, "
            f"but only {len(valid_indices)} valid starting "
            f"steps exist."
        )

    # -----------------------------------------------------------------------
    # Select evaluation episodes deterministically.
    # -----------------------------------------------------------------------
    g = np.random.default_rng(
        cfg.seed
    )

    chosen_sub_indices = g.choice(
        len(valid_indices),
        size=cfg.eval.num_eval,
        replace=False,
    )

    random_episode_indices = np.sort(
        valid_indices[chosen_sub_indices]
    )

    # -----------------------------------------------------------------------
    # Retrieve selected dataset rows.
    # -----------------------------------------------------------------------
    selected_rows = dataset.get_row_data(
        random_episode_indices
    )

    eval_episodes = selected_rows[
        col_name
    ]

    eval_start_idx = selected_rows[
        "step_idx"
    ]

    # -----------------------------------------------------------------------
    # Attach policy to environment.
    # -----------------------------------------------------------------------
    world.set_policy(policy)

    # -----------------------------------------------------------------------
    # Create output directory.
    # -----------------------------------------------------------------------
    results_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Optional callbacks.
    # -----------------------------------------------------------------------
    callables_cfg = cfg.eval.get(
        "callables"
    )

    callables = (
        OmegaConf.to_container(
            callables_cfg,
            resolve=True,
        )
        if callables_cfg
        else None
    )

    # -----------------------------------------------------------------------
    # Run evaluation.
    # -----------------------------------------------------------------------
    print("Starting evaluation...")

    start_time = time.time()

    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=callables,
        video=results_path,
    )

    end_time = time.time()

    # -----------------------------------------------------------------------
    # Print metrics.
    # -----------------------------------------------------------------------
    print(metrics)

    # -----------------------------------------------------------------------
    # Save results.
    # -----------------------------------------------------------------------
    output_file = (
        results_path
        / cfg.output.filename
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open("a") as f:

        f.write("\n==== CONFIG ====\n")
        f.write(
            OmegaConf.to_yaml(cfg)
        )

        f.write("\n==== RESULTS ====\n")
        f.write(
            f"metrics: {metrics}\n"
        )

        f.write(
            "evaluation_time: "
            f"{end_time - start_time:.4f} seconds\n"
        )


if __name__ == "__main__":
    run()
