import os
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


# ---------------------------------------------------------------------------
# Explicit gradient planner (Sequential / One-by-One)
# ---------------------------------------------------------------------------
class ExplicitGradientSolver:
    """Gradient-descent planner running per-episode sequential optimization."""

    def __init__(
        self,
        model,
        n_steps: int,
        device: str | torch.device = "cuda",
        lr: float = 1.0,
        action_noise: float = 0.0,
        grad_clip: float | None = None,
        seed: int = 1234,
        process: dict | None = None,
        action_bounds=None,
        action_block: int = 5,
        horizon: int = 5,
        raw_action_dim: int = 2,
    ):
        self.model = model
        self.n_steps = int(n_steps)
        self.device = torch.device(device)
        self.lr = float(lr)
        self.action_noise = float(action_noise)
        self.grad_clip = grad_clip
        self.process = process or {}
        self.action_bounds = action_bounds

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._action_block = int(action_block)
        self._horizon = int(horizon)
        self._single_action_dim = int(raw_action_dim)

        action_encoder = getattr(self.model, "action_encoder", None)
        encoder_in_dim = getattr(action_encoder, "in_features", None) if action_encoder is not None else None

        if encoder_in_dim is not None:
            self._action_dim = int(encoder_in_dim)
        else:
            self._action_dim = self._single_action_dim * self._action_block

        self._n_envs = None
        self._configured = True
        self._dtype = torch.float32

        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            pass

        print(
            f"ExplicitGradientSolver initialized (Sequential Mode): raw_action_dim={self._single_action_dim}, "
            f"action_block={self._action_block}, "
            f"optimized_action_dim={self._action_dim}, "
            f"horizon={self._horizon}"
        )

    def configure(self, *, action_space=None, n_envs: int = 1, config=None, **kwargs):
        self._configured = True
        if n_envs is not None:
            self._n_envs = int(n_envs)
        if config is not None:
            if hasattr(config, "action_block"):
                self._action_block = int(config.action_block)
            if hasattr(config, "horizon"):
                self._horizon = int(config.horizon)

        if action_space is not None and hasattr(action_space, "shape") and len(action_space.shape) > 0:
            env_dim = int(np.prod(action_space.shape))
            if env_dim <= 10:
                self._single_action_dim = env_dim

        action_encoder = getattr(self.model, "action_encoder", None)
        encoder_in_dim = getattr(action_encoder, "in_features", None) if action_encoder is not None else None
        if encoder_in_dim is not None:
            self._action_dim = int(encoder_in_dim)
        else:
            self._action_dim = self._single_action_dim * self._action_block

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def horizon(self):
        return self._horizon

    @property
    def dtype(self):
        return self._dtype

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _first_tensor(info_dict: dict) -> torch.Tensor:
        for value in info_dict.values():
            if torch.is_tensor(value):
                return value
        raise ValueError("info_dict contains no tensor values")

    def _move_to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(self.device)
        return value

    def _slice_info_dict(self, info_dict: dict, idx: int) -> dict:
        """Extract a single episode context (batch size = 1) from the batch info_dict."""
        sliced = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                sliced[k] = v[idx : idx + 1]
            elif isinstance(v, (list, tuple)) and len(v) > idx:
                sliced[k] = [v[idx]]
            else:
                sliced[k] = v
        return sliced

    def _adapt_action_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected action tensor with shape (B, D) or (B, T, D), got {tuple(x.shape)}")

        current_feat = x.shape[-1]
        target_feat = self._action_dim

        if current_feat == target_feat:
            return x

        if target_feat % current_feat == 0:
            repeat_factor = target_feat // current_feat
            return x.repeat(1, 1, repeat_factor)

        if current_feat > target_feat:
            return x[..., :target_feat]
        else:
            pad = torch.zeros(*x.shape[:-1], target_feat - current_feat, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=-1)

    def _build_context(self, info_dict: dict) -> dict:
        context = {}
        ignored = {
            "goal",
            "goal_pixels",
            "terminated",
            "truncated",
            "_needs_flush",
            "action_history",
        }

        for key, value in info_dict.items():
            if key not in ignored:
                context[key] = self._move_to_device(value)

        if "action_history" in info_dict:
            hist = self._move_to_device(info_dict["action_history"])
            hist = self._adapt_action_tensor(hist)

            current = info_dict.get("action")
            if current is not None and torch.is_tensor(current):
                current = self._adapt_action_tensor(current)
                context["action"] = torch.cat([hist, current], dim=1)
            else:
                context["action"] = hist

        elif "action" in info_dict and torch.is_tensor(info_dict["action"]):
            context["action"] = self._adapt_action_tensor(info_dict["action"])

        return context

    def _build_goal(self, info_dict: dict) -> dict:
        if "goal" not in info_dict:
            raise KeyError("Evaluation info_dict must contain a 'goal' key")

        goal = {
            "pixels": self._move_to_device(info_dict["goal"]),
        }

        for key, value in info_dict.items():
            if key.startswith("goal_") and key != "goal_pixels":
                goal[key] = self._move_to_device(value)

        return goal

    def _normalized_bounds(self):
        if self.action_bounds is None or "action" not in self.process:
            return None

        scaler = self.process["action"]
        scaler_dim = int(scaler.mean_.shape[0])

        raw_lo, raw_hi = self.action_bounds
        raw_lo = np.asarray(raw_lo, dtype=np.float32)
        raw_hi = np.asarray(raw_hi, dtype=np.float32)

        if scaler_dim == self._single_action_dim:
            if raw_lo.ndim == 0:
                raw_lo = np.full((self._single_action_dim,), raw_lo.item(), dtype=np.float32)
            if raw_hi.ndim == 0:
                raw_hi = np.full((self._single_action_dim,), raw_hi.item(), dtype=np.float32)

            lo = scaler.transform(raw_lo.reshape(1, -1))[0]
            hi = scaler.transform(raw_hi.reshape(1, -1))[0]

            repeat_factor = max(1, self._action_dim // self._single_action_dim)
            lo = np.tile(lo, repeat_factor)[: self._action_dim]
            hi = np.tile(hi, repeat_factor)[: self._action_dim]
            return torch.as_tensor(lo, device=self.device, dtype=self.dtype), torch.as_tensor(
                hi, device=self.device, dtype=self.dtype
            )

        return None

    def _initial_action(self, batch_size: int, init_action: torch.Tensor | None):
        if init_action is None:
            actions = torch.randn(
                batch_size,
                self.horizon,
                self.action_dim,
                device=self.device,
                dtype=self.dtype,
                generator=self.generator,
                requires_grad=True,
            )
        else:
            actions = init_action.to(device=self.device, dtype=self.dtype).clone().detach()

            if actions.ndim == 4:
                actions = actions[:, 0]

            if actions.shape[-1] != self.action_dim:
                actions = self._adapt_action_tensor(actions)

            if actions.shape[1] < self.horizon:
                pad = torch.zeros(
                    batch_size,
                    self.horizon - actions.shape[1],
                    self.action_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                actions = torch.cat([actions, pad], dim=1)
            elif actions.shape[1] > self.horizon:
                actions = actions[:, : self.horizon]

            actions.requires_grad_(True)

        bounds = self._normalized_bounds()
        if bounds is not None:
            lo, hi = bounds
            with torch.no_grad():
                actions.clamp_(lo, hi)

        return actions

    # ---------------------------------------------------------------------
    # Latent objective for a single sequence
    # ---------------------------------------------------------------------
    def _latent_energy(self, context_data: dict, goal_data: dict, act_seq: torch.Tensor):
        with torch.no_grad():
            ctx_output = self.model.encode(context_data)
            ctx_emb = ctx_output["emb"]
            ctx_act = ctx_output["act_emb"]

            goal_output = self.model.encode(goal_data)
            goal_emb = goal_output["emb"][:, -1]

        act_emb_seq = self.model.action_encoder(act_seq)

        current_ctx_emb = ctx_emb
        current_ctx_act = ctx_act
        final_pred_emb = None

        for t in range(self.horizon):
            step_act_emb = act_emb_seq[:, t : t + 1]

            full_act_ctx = torch.cat(
                [current_ctx_act[:, 1:], step_act_emb],
                dim=1,
            )

            pred_out = self.model.predict(current_ctx_emb, full_act_ctx)
            pred_emb = pred_out[:, -1] if pred_out.dim() == 3 else pred_out

            current_ctx_emb = torch.cat(
                [current_ctx_emb[:, 1:], pred_emb.unsqueeze(1)],
                dim=1,
            )
            current_ctx_act = full_act_ctx
            final_pred_emb = pred_emb

        sq_error = (final_pred_emb - goal_emb.detach()).pow(2)
        return sq_error.mean()

    # ---------------------------------------------------------------------
    # Sequential Optimizer
    # ---------------------------------------------------------------------
    def _optimize_single_episode(
        self,
        ep_info: dict,
        ep_init_action: torch.Tensor | None,
        bounds: tuple | None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run gradient descent planning for one single episode."""
        context = self._build_context(ep_info)
        goal = self._build_goal(ep_info)
        actions = self._initial_action(batch_size=1, init_action=ep_init_action)

        energy_history = []

        for _ in range(self.n_steps):
            energy = self._latent_energy(context, goal, actions)

            grad_energy = torch.autograd.grad(
                energy,
                actions,
                create_graph=False,
                retain_graph=False,
            )[0]

            energy_history.append(float(energy.detach().cpu().item()))

            with torch.no_grad():
                if self.grad_clip is not None:
                    grad_norm = grad_energy.norm()
                    if grad_norm > self.grad_clip:
                        grad_energy = grad_energy * (self.grad_clip / (grad_norm + 1e-6))

                actions -= self.lr * grad_energy

                if self.action_noise > 0.0:
                    actions += self.action_noise * torch.randn(
                        actions.shape,
                        device=self.device,
                        dtype=self.dtype,
                        generator=self.generator,
                    )

                if bounds is not None:
                    lo, hi = bounds
                    actions.clamp_(lo, hi)

            actions.requires_grad_(True)

        final_energy = self._latent_energy(context, goal, actions)
        energy_history.append(float(final_energy.detach().cpu().item()))

        return actions.detach().cpu(), energy_history

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()

        first = self._first_tensor(info_dict)
        batch_size = len(first)

        bounds = self._normalized_bounds()

        # Parse episode identifiers for logging
        episode_ids = None
        for key in ("episode_idx", "ep_idx"):
            if key in info_dict:
                value = info_dict[key]
                if torch.is_tensor(value):
                    episode_ids = value.detach().cpu().reshape(-1).tolist()
                elif isinstance(value, np.ndarray):
                    episode_ids = value.reshape(-1).tolist()
                elif isinstance(value, (list, tuple)):
                    episode_ids = list(value)
                break

        gathered_actions = []
        all_final_mses = []
        full_cost_traces = []

        print(f"\nRunning sequential optimization across {batch_size} episodes...")

        for i in range(batch_size):
            ep_info = self._slice_info_dict(info_dict, i)
            ep_init = init_action[i : i + 1] if init_action is not None else None

            opt_action, cost_trace = self._optimize_single_episode(ep_info, ep_init, bounds)

            gathered_actions.append(opt_action)
            all_final_mses.append(cost_trace[-1])
            full_cost_traces.append(cost_trace)

            ep_label = (
                episode_ids[i]
                if episode_ids is not None and i < len(episode_ids)
                else i
            )
            print(f"  [Episode {ep_label}] final_MSE={cost_trace[-1]:.8f}")

        # Stack batch back into (B, horizon, action_dim) for WorldModelPolicy
        actions_out = torch.cat(gathered_actions, dim=0)

        mean_cost_trace = np.mean(full_cost_traces, axis=0).tolist()
        elapsed = time.time() - start_time

        print(f"Batch mean final MSE: {np.mean(all_final_mses):.8f}")
        print(f"ExplicitGradientSolver.solve completed sequentially in {elapsed:.4f}s\n")

        return {
            "actions": actions_out,
            "cost": mean_cost_trace,
        }


# ---------------------------------------------------------------------------
# Standard preprocessing / dataset helpers
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run PushT evaluation with explicit latent gradient descent planning."""

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue

        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy_name = cfg.get("policy", "random")

    if policy_name == "random":
        policy = swm.policy.RandomPolicy()
    else:
        ckpt_path = policy_name
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path += ".ckpt"

        print(f"Loading local PyTorch model from {ckpt_path}...")
        model = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg.plan_config)

        solver_cfg = cfg.get("solver", cfg.get("gradient_solver", {}))
        optimizer_kwargs = solver_cfg.get("optimizer_kwargs", {})
        lr = float(optimizer_kwargs.get("lr", 1.0))

        explicit_solver = ExplicitGradientSolver(
            model=model,
            n_steps=int(solver_cfg.get("n_steps", 50)),
            device=device,
            lr=lr,
            action_noise=float(solver_cfg.get("action_noise", 0.0)),
            grad_clip=solver_cfg.get("grad_clip", None),
            seed=int(cfg.get("seed", 1234)),
            process=process,
            action_bounds=solver_cfg.get("action_bounds", None),
            action_block=int(cfg.plan_config.action_block),
            horizon=int(cfg.plan_config.horizon),
            raw_action_dim=2,
        )

        policy = swm.policy.WorldModelPolicy(
            solver=explicit_solver,
            config=config,
            process=process,
            transform=transform,
        )

    results_path = (
        Path(swm.data.utils.get_cache_dir(), policy_name).parent
        if policy_name != "random"
        else Path(__file__).parent
    )

    # ------------------------------------------------------------------
    # Select valid evaluation starting points
    # ------------------------------------------------------------------
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }

    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1,
        size=cfg.eval.num_eval,
        replace=False,
    )

    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print(random_episode_indices)

    eval_rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = eval_rows[col_name]
    eval_start_idx = eval_rows["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    # ------------------------------------------------------------------
    # Evaluate with the normal stable-worldmodel environment loop
    # ------------------------------------------------------------------
    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get("callables"),
            resolve=True,
        ),
        video=results_path,
    )
    end_time = time.time()

    print(metrics)

    output_path = results_path / cfg.output.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        f.write(
            "planner: ExplicitGradientSolver (Sequential); "
            "objective: mean((final_pred_emb - goal_emb)^2); "
            "gradient: torch.autograd.grad\n"
        )


if __name__ == "__main__":
    run()
