import os
os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from stable_worldmodel.solver.gd import GradientSolver

# ---------------------------------------------------------------------------
# PATCH: stable-worldmodel 0.1.1 bug — leave the warm-start action tensor on CPU
# ---------------------------------------------------------------------------
def _patched_init_action(self, n_envs, actions=None):
    if actions is None:
        actions = torch.zeros((n_envs, 0, self.action_dim), dtype=self.dtype)

    remaining = self.horizon - actions.shape[1]
    if remaining > 0:
        new_actions = torch.zeros(n_envs, remaining, self.action_dim, dtype=self.dtype)
        actions = torch.cat([actions, new_actions], dim=1)

    actions = actions.to(self.device)  # <-- always move, not just when padding was needed

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


# ---------------------------------------------------------------------------
# CUSTOM SOLVER: Forces Sum of Squared Errors (SSE) instead of MSE 
# during the gradient descent planning loop.
# ---------------------------------------------------------------------------
_orig_mse_loss = F.mse_loss
_orig_mse_module_init = nn.MSELoss.__init__
_orig_tensor_mean = torch.Tensor.mean
_orig_torch_mean = torch.mean

def _sse_functional(*args, **kwargs):
    # Only override if reduction is 'mean' or not specified.
    # If the model requests 'none', we leave it alone to prevent shape crashes.
    if kwargs.get("reduction", "mean") != "none":
        kwargs["reduction"] = "sum"
    return _orig_mse_loss(*args, **kwargs)

def _sse_module_init(self, *args, **kwargs):
    if kwargs.get("reduction", "mean") != "none":
        kwargs["reduction"] = "sum"
    _orig_mse_module_init(self, *args, **kwargs)

def _sum_mean_no_dim(self, *args, **kwargs):
    # If .mean() is called without dimensions (e.g. to get a scalar loss),
    # replace it with .sum() to avoid averaging.
    if not args and not kwargs:
        return self.sum()
    return _orig_tensor_mean(self, *args, **kwargs)

def _sum_torch_mean_no_dim(input, *args, **kwargs):
    # Same patch for the functional torch.mean(tensor) version
    if not args and not kwargs:
        return input.sum()
    return _orig_torch_mean(input, *args, **kwargs)

class CustomSolver(GradientSolver):
    def solve(self, data, init_action=None):
        """
        Wraps the original solve method. We patch PyTorch's MSE and mean functions
        to use 'sum' reduction right before calling the library's solve logic,
        and restore them immediately after.
        """
        # 1. Patch the global MSE and mean functions
        F.mse_loss = _sse_functional
        nn.MSELoss.__init__ = _sse_module_init
        torch.Tensor.mean = _sum_mean_no_dim
        torch.mean = _sum_torch_mean_no_dim
        
        # 2. Replace any pre-existing nn.MSELoss instances on this solver
        replaced_modules = {}
        for attr_name in list(vars(self)):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, nn.MSELoss):
                replaced_modules[attr_name] = attr
                setattr(self, attr_name, nn.MSELoss(reduction="sum"))
                
        try:
            # 3. Call the original, library-provided solve method
            return super().solve(data, init_action=init_action)
        finally:
            # 4. Restore everything to avoid side effects on future calls
            F.mse_loss = _orig_mse_loss
            nn.MSELoss.__init__ = _orig_mse_module_init
            torch.Tensor.mean = _orig_tensor_mean
            torch.mean = _orig_torch_mean
            for attr_name, attr in replaced_modules.items():
                setattr(self, attr_name, attr)
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
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
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

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        
        ckpt_path = cfg.policy 
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path += ".ckpt"  # Ensure it looks for a .ckpt file
            
        print(f"Loading local PyTorch model from {ckpt_path}...")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)        
        
        # --- FIX 1: Explicitly move the model to the GPU ---
        model = model.to("cuda")
        
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        
        # Instantiate the original solver via Hydra to ensure all configs are populated
        solver = hydra.utils.instantiate(cfg.solver, model=model, device="cuda")
        
        # Swap the class to our CustomSolver so it uses our SSE wrapper
        solver.__class__ = CustomSolver
                    
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

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

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
        
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
