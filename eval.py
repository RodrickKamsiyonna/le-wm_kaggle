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
# ---------------------------------------------------------------------------
def _patched_init_action(self, n_envs, actions=None):
    if actions is None:
        actions = torch.zeros((n_envs, 0, self.action_dim), dtype=self.dtype)

    remaining = self.horizon - actions.shape[1]
    if remaining > 0:
        new_actions = torch.zeros(n_envs, remaining, self.action_dim, dtype=self.dtype)
        actions = torch.cat([actions, new_actions], dim=1)

    actions = actions.to(self.device)
    actions = actions.unsqueeze(1).repeat_interleave(self.num_samples, dim=1)
    if hasattr(self, "init") and self.init.shape == actions.shape:
        self.init.copy_(actions)
    else:
        if "init" in self._parameters:
            del self._parameters["init"]
        self.register_parameter("init", torch.nn.Parameter(actions))


GradientSolver.init_action = _patched_init_action
# ---------------------------------------------------------------------------


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = [np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes]
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # Create transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy_name = cfg.get("policy", "random")

    if policy_name != "random":
        ckpt_path = policy_name
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path += ".ckpt"

        print(f"Loading local PyTorch model from {ckpt_path}...")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model, device=device)

        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
        )
    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if policy_name != "random"
        else Path(__file__).resolve().parent
    )

    # Filter for valid start steps given the evaluation goal offset
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    all_row_ep_indices = dataset.get_col_data(col_name)
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in all_row_ep_indices])

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{len(valid_indices)} valid starting points found for evaluation.")

    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Requested {cfg.eval.num_eval} evaluations, but only {len(valid_indices)} valid starting steps exist."
        )

    g = np.random.default_rng(cfg.seed)
    chosen_sub_indices = g.choice(len(valid_indices), size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[chosen_sub_indices])

    selected_rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = selected_rows[col_name]
    eval_start_idx = selected_rows["step_idx"]

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    callables_cfg = cfg.eval.get("callables")
    callables = OmegaConf.to_container(callables_cfg, resolve=True) if callables_cfg else None

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

    print(metrics)

    output_file = results_path / cfg.output.filename
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time:.4f} seconds\n")


if __name__ == "__main__":
    run()
