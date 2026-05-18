import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


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


class GradientBasedSolver:
    """
    Gradient-based action sequence optimizer.

    Instead of sampling action candidates (CEM/random-shooting), this solver
    treats the action sequence as a continuous variable and minimizes the
    world-model energy — the squared prediction error between the rolled-out
    latent trajectory and the goal latent — via gradient descent.

    This mirrors the EQM training objective in lejepa_forward: the model was
    trained so that grad_a E(s, a, s') points toward better actions, so at
    eval time we can follow exactly that gradient.

    Args:
        model:          AutoCostModel (encoder + predictor + action_encoder).
        action_dim:     Dimensionality of a single action vector.
        horizon:        Number of action steps to optimize.
        action_block:   How many optimized steps to actually execute before
                        re-planning (receding horizon).
        n_iter:         Number of gradient-descent iterations per planning call.
        lr:             Learning rate for the action optimizer.
        n_restarts:     Number of random restarts; best solution is kept.
        action_noise:   Std of Gaussian noise added to actions after every
                        gradient step (exploration / avoid local minima).
        action_bounds:  Optional (low, high) tuple to clamp actions each step.
    """

    def __init__(
        self,
        model,
        action_dim: int,
        horizon: int,
        action_block: int,
        n_iter: int = 50,
        lr: float = 0.05,
        n_restarts: int = 4,
        action_noise: float = 0.0,
        action_bounds: tuple | None = None,
    ):
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_block = action_block
        self.n_iter = n_iter
        self.lr = lr
        self.n_restarts = n_restarts
        self.action_noise = action_noise
        self.action_bounds = action_bounds  # e.g. (-1.0, 1.0)

    @torch.no_grad()
    def _encode_context(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the current observation context into latent embeddings."""
        output = self.model.encode(batch)
        # output["emb"]: (1, ctx_len, hidden_dim)
        # output["act_emb"]: (1, ctx_len, embed_dim)
        return output["emb"], output["act_emb"]

    def _compute_energy(
        self,
        ctx_emb: torch.Tensor,      # (1, ctx_len, hidden_dim)
        act_seq: torch.Tensor,       # (1, horizon, action_dim)  — differentiable
        goal_emb: torch.Tensor,      # (1, hidden_dim)
    ) -> torch.Tensor:
        """
        Roll out the predictor for `horizon` steps with act_seq and return the
        total squared distance to the goal latent.

        Energy = sum_{t=1}^{horizon} ||pred_emb_t - goal_emb||^2

        Because act_seq.requires_grad=True and we use only differentiable ops,
        autograd can trace back through this energy to act_seq.
        """
        # Encode the candidate action sequence through the action encoder
        # act_seq: (1, horizon, action_dim)  →  act_emb_seq: (1, horizon, embed_dim)
        act_emb_seq = self.model.action_encoder(act_seq)

        current_ctx = ctx_emb          # (1, ctx_len, hidden_dim)
        current_act = act_emb_seq      # (1, horizon, embed_dim)

        total_energy = torch.tensor(0.0, device=ctx_emb.device, dtype=ctx_emb.dtype)

        for t in range(self.horizon):
            # Predict next latent from current context + current action embedding
            # predict expects (B, ctx_len, ...) — pass the single action at step t
            pred_emb = self.model.predict(current_ctx, current_act[:, t : t + 1])
            # pred_emb: (1, 1, hidden_dim) or (1, hidden_dim) depending on impl
            pred_emb = pred_emb.squeeze(1)   # → (1, hidden_dim)

            # Energy: squared distance to goal
            total_energy = total_energy + F.mse_loss(pred_emb, goal_emb.squeeze(1))

            # Slide context window: drop oldest, append prediction
            current_ctx = torch.cat(
                [current_ctx[:, 1:], pred_emb.unsqueeze(1)], dim=1
            )

        return total_energy

    def solve(
        self,
        batch: dict,
        goal_emb: torch.Tensor,     # (1, hidden_dim)  goal latent from encoder
    ) -> torch.Tensor:
        """
        Return the best action block (action_block, action_dim) found by
        gradient descent over `n_restarts` random initializations.
        """
        device = goal_emb.device
        dtype = goal_emb.dtype

        ctx_emb, _ = self._encode_context(batch)  # (1, ctx_len, hidden_dim)

        best_energy = float("inf")
        best_actions = None

        for _ in range(self.n_restarts):
            # Random initialization of action sequence
            act_seq = torch.randn(
                1, self.horizon, self.action_dim, device=device, dtype=dtype
            )
            if self.action_bounds is not None:
                lo, hi = self.action_bounds
                act_seq = act_seq.clamp(lo, hi)
            act_seq = act_seq.requires_grad_(True)

            optimizer = torch.optim.Adam([act_seq], lr=self.lr)

            for _ in range(self.n_iter):
                optimizer.zero_grad()
                energy = self._compute_energy(ctx_emb, act_seq, goal_emb)
                energy.backward()
                optimizer.step()

                # Optional: additive exploration noise after each update
                with torch.no_grad():
                    if self.action_noise > 0.0:
                        act_seq.data += self.action_noise * torch.randn_like(act_seq)
                    if self.action_bounds is not None:
                        lo, hi = self.action_bounds
                        act_seq.data.clamp_(lo, hi)

            final_energy = self._compute_energy(ctx_emb, act_seq, goal_emb).item()
            if final_energy < best_energy:
                best_energy = final_energy
                best_actions = act_seq.detach().clone()  # (1, horizon, action_dim)

        # Return only the first action_block steps, squeezed to (action_block, action_dim)
        return best_actions[0, : self.action_block]


class GradientWorldModelPolicy:
    """
    Wraps GradientBasedSolver in the same interface as WorldModelPolicy so
    it can be passed to world.set_policy() unchanged.

    The key difference from the CEM-based WorldModelPolicy:
      - No population of random samples; a single (or restarted) sequence is
        optimized via backprop through the world model.
      - The gradient signal comes from the EQM-trained energy landscape, so
        the planner is consistent with how the model was trained.
    """

    def __init__(
        self,
        solver: GradientBasedSolver,
        config,          # swm.PlanConfig
        process: dict,   # sklearn scalers for normalization
        transform: dict, # image transforms
    ):
        self.solver = solver
        self.config = config
        self.process = process
        self.transform = transform

    def _preprocess_obs(self, obs: dict) -> dict:
        """Apply normalization and image transforms to a raw observation."""
        batch = {}
        for key, val in obs.items():
            if key in self.transform:
                batch[key] = self.transform[key](val).unsqueeze(0)  # (1, C, H, W)
            elif key in self.process:
                arr = np.array(val, dtype=np.float32).reshape(1, -1)
                batch[key] = torch.tensor(
                    self.process[key].transform(arr), dtype=torch.float32
                )
            else:
                batch[key] = torch.tensor(np.array(val), dtype=torch.float32).unsqueeze(0)
        return batch

    @torch.no_grad()
    def _encode_goal(self, goal_obs: dict) -> torch.Tensor:
        """Encode the goal observation into a latent vector."""
        batch = self._preprocess_obs(goal_obs)
        batch = {k: v.to(next(self.solver.model.parameters()).device) for k, v in batch.items()}
        output = self.solver.model.encode(batch)
        # Use the last frame embedding as the goal
        return output["emb"][:, -1:]   # (1, 1, hidden_dim)

    def act(self, obs: dict, goal_obs: dict) -> np.ndarray:
        """
        Given current observation and goal observation, return the next
        action_block actions as a numpy array of shape (action_block, action_dim).
        """
        device = next(self.solver.model.parameters()).device

        batch = self._preprocess_obs(obs)
        batch = {k: v.to(device) for k, v in batch.items()}

        goal_emb = self._encode_goal(goal_obs)   # (1, 1, hidden_dim)

        # solve returns (action_block, action_dim) tensor
        actions = self.solver.solve(batch, goal_emb)

        # Denormalize actions back to environment scale
        actions_np = actions.cpu().numpy()  # (action_block, action_dim)
        if "action" in self.process:
            actions_np = self.process["action"].inverse_transform(actions_np)

        return actions_np


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation using gradient-based action planning."""
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

    # -- run evaluation
    policy_name = cfg.get("policy", "random")

    if policy_name != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg.plan_config)

        # ── Gradient-based solver (replaces CEM/random-shooting solver) ──────
        grad_cfg = cfg.get("gradient_solver", {})
        solver = GradientBasedSolver(
            model=model,
            action_dim=cfg.wm.action_dim,
            horizon=cfg.plan_config.horizon,
            action_block=cfg.plan_config.action_block,
            n_iter=grad_cfg.get("n_iter", 50),
            lr=grad_cfg.get("lr", 0.05),
            n_restarts=grad_cfg.get("n_restarts", 4),
            action_noise=grad_cfg.get("action_noise", 0.0),
            action_bounds=tuple(grad_cfg.action_bounds) if grad_cfg.get("action_bounds") else None,
        )

        policy = GradientWorldModelPolicy(
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
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
