import os
import multiprocessing as mp

# Set multiprocessing start method to 'spawn' before any CUDA/Lance/Gym calls
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from stable_worldmodel.solver.gd import GradientSolver

# ---------------------------------------------------------------------------
# PATCH: stable-worldmodel 0.1.1 bug — GradientSolver.init_action()
# Ensure all action tensors and random perturbations are explicitly placed
# on self.device to avoid CPU/CUDA device mismatches on the initial call.
# ---------------------------------------------------------------------------
def _patched_init_action(self, n_envs, actions=None):
    if actions is None:
        actions = torch.zeros((n_envs, 0, self.action_dim), dtype=self.dtype)

    remaining = self.horizon - actions.shape[1]
    if remaining > 0:
        new_actions = torch.zeros(n_envs, remaining, self.action_dim, dtype=self.dtype)
        actions = torch.cat([actions, new_actions], dim=1)

    actions = actions.to(self.device)  # Always ensure tensor is moved to target device

    actions = actions.unsqueeze(1).repeat_interleave(self.num_samples, dim=1)
    actions[:, 1:] += (
        torch.randn(
            actions[:, 1:].shape,
            generator=self.torch_gen,
            device=self.device,
            dtype=self.dtype,
        )
        * self.var_scale
    )

    if hasattr(self, "init") and self.init.shape == actions.shape:
        self.init.copy_(actions)
    else:
        if "init" in self._parameters:
            del self._parameters["init"]
        self.register_parameter("init", torch.nn.Parameter(actions))


GradientSolver.init_action = _patched_init_action
# ---------------------------------------------------------------------------


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of learned world model vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # Create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # Create image transforms
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    # Load dataset and calculate normalization statistics
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

    # Instantiate policy
    policy = cfg.get("policy", "random")

    if policy != "random":
        ckpt_path = cfg.policy
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path += ".ckpt"

        print(f"Loading local PyTorch model from {ckpt_path}...")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Move model to CUDA and set evaluation modes
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model, device="cuda")

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
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # Sample starting indices for evaluation episodes
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # Filter for valid step starting points
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # Sort to optimize HDF5 slice access
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print("Sampled episode indices:", random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    # Execute evaluation loop
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()

    print("\n==== Evaluation Complete ====")
    print("Metrics:", metrics)
    print(f"Total time: {end_time - start_time:.2f} seconds")

    # Save output artifacts
    results_file = results_path / cfg.output.filename
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time:.2f} seconds\n")


if __name__ == "__main__":
    run()
