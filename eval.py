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
# Explicit gradient planner
# ---------------------------------------------------------------------------
# This deliberately mirrors the optimization code from the supplied diagram:
#
#   act_emb_seq = model.action_encoder(act_seq)
#   rollout through model.predict(...)
#   energy = (final_pred_emb - goal_emb).pow(2).mean()
#   grad_energy = torch.autograd.grad(energy, act_seq)[0]
#   act_seq -= lr * grad_energy
#
# There is NO optimizer object, NO model.get_cost(), and NO .backward().
# The world-model parameters stay frozen, but gradients are still propagated
# through the frozen model with respect to the action sequence.
# ---------------------------------------------------------------------------
class ExplicitGradientSolver:
    """Gradient-descent planner using an explicit latent MSE objective."""

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
        self._configured = False
        self._n_envs = None
        self._action_dim = None
        self._action_block = 1
        self._horizon = None
        self._dtype = torch.float32

        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            pass

    def configure(self, *, action_space, n_envs: int, config):
        self._configured = True
        self._n_envs = int(n_envs)
        self._action_block = int(config.action_block)
        self._horizon = int(config.horizon)

        # IMPORTANT: Do NOT infer the raw action dimension from
        # process["action"] here. In this evaluation setup the cached dataset
        # action column can already contain a flattened/chunked representation
        # (e.g. 100 features), while the environment still emits the true
        # PushT action (x, y) = 2 features.
        #
        # The authoritative dimension for the optimizer is the input dimension
        # of the model's action encoder. For action_block=5 and PushT, the JEPA
        # action encoder expects 10 features, hence the raw action dimension is
        # 10 / 5 = 2.
        encoder_in_dim = None
        action_encoder = getattr(self.model, "action_encoder", None)
        if action_encoder is not None:
            encoder_in_dim = getattr(action_encoder, "in_features", None)

        if encoder_in_dim is not None:
            encoder_in_dim = int(encoder_in_dim)
            if encoder_in_dim % self._action_block != 0:
                raise ValueError(
                    f"Model action_encoder.in_features={encoder_in_dim} is not "
                    f"divisible by action_block={self._action_block}."
                )
            self._single_action_dim = encoder_in_dim // self._action_block
        else:
            # Fallback only when the action encoder does not expose in_features.
            env_dim = int(np.prod(action_space.shape))
            self._single_action_dim = env_dim

        self._action_dim = self._single_action_dim * self._action_block

        print(
            f"ExplicitGradientSolver configured: raw_action_dim={self._single_action_dim}, "
            f"action_block={self._action_block}, "
            f"optimized_action_dim={self._action_dim}, "
            f"horizon={self._horizon}"
        )

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

    def _build_context(self, info_dict: dict) -> dict:
        """Build the context dictionary used by model.encode()."""
        context = {}

        # Keep the same observation information used by the supplied diagram.
        # Control/termination metadata should not be fed into the encoder.
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

        # The JEPA action encoder expects CHUNKED actions:
        # raw PushT action (..., 2), action_block=5 -> encoder input (..., 10).
        def chunk_current_action(x: torch.Tensor) -> torch.Tensor:
            x = x.to(self.device)

            if x.ndim == 2:
                x = x.unsqueeze(1)  # (B, D) -> (B, 1, D)
            elif x.ndim != 3:
                raise ValueError(
                    f"Expected current action tensor with shape (B, D) or (B, T, D), "
                    f"got {tuple(x.shape)}"
                )

            raw_dim = self._single_action_dim
            chunked_dim = raw_dim * self._action_block

            if x.shape[-1] == chunked_dim:
                return x

            if x.shape[-1] != raw_dim:
                raise ValueError(
                    f"Unexpected current action dimension {x.shape[-1]}. "
                    f"Expected raw dim {raw_dim} or chunked dim {chunked_dim}."
                )

            # Same operation as the explicit diagram: t.repeat(1, action_chunk).
            return x.repeat(1, 1, self._action_block)

        if "action_history" in info_dict:
            # stable-worldmodel supplies action_history already in solver space
            # (chunked per action block). Do NOT repeat it again.
            hist = self._move_to_device(info_dict["action_history"])
            if hist.ndim == 2:
                hist = hist.unsqueeze(1)
            if hist.ndim != 3:
                raise ValueError(
                    f"Expected action_history with shape (B, T, D), got {tuple(hist.shape)}"
                )

            expected_dim = self._single_action_dim * self._action_block
            if hist.shape[-1] != expected_dim:
                raise ValueError(
                    f"Unexpected action_history dimension {hist.shape[-1]}. "
                    f"Expected solver/chunked dim {expected_dim}."
                )

            current = info_dict.get("action")
            if current is not None and torch.is_tensor(current):
                current = chunk_current_action(current)
                context["action"] = torch.cat([hist, current], dim=1)
            else:
                context["action"] = hist

        elif "action" in info_dict and torch.is_tensor(info_dict["action"]):
            context["action"] = chunk_current_action(info_dict["action"])

        return context

    def _build_goal(self, info_dict: dict) -> dict:
        """Build the goal dictionary used by model.encode()."""
        if "goal" not in info_dict:
            raise KeyError("Evaluation info_dict must contain a 'goal' key")

        goal = {
            "pixels": self._move_to_device(info_dict["goal"]),
        }

        # Preserve any explicit goal-side auxiliary features if present.
        for key, value in info_dict.items():
            if key.startswith("goal_") and key != "goal_pixels":
                goal[key] = self._move_to_device(value)

        return goal

    def _normalized_bounds(self):
        """Convert raw action bounds to the same normalized action space as process['action']."""
        if self.action_bounds is None or "action" not in self.process:
            return None

        # Bounds are only valid when the scaler itself represents the raw
        # action dimension. Some PushT datasets store a flattened historical
        # action representation with many more features, so do not attempt to
        # apply those bounds in that case.
        scaler_dim = int(self.process["action"].mean_.shape[0])
        if scaler_dim != self._single_action_dim:
            print(
                f"Skipping action bounds: scaler action dim={scaler_dim}, "
                f"but raw PushT action dim={self._single_action_dim}."
            )
            return None

        raw_lo, raw_hi = self.action_bounds
        raw_lo = np.asarray(raw_lo, dtype=np.float32)
        raw_hi = np.asarray(raw_hi, dtype=np.float32)

        if raw_lo.ndim == 0:
            raw_lo = np.full((self._single_action_dim,), raw_lo.item(), dtype=np.float32)
        if raw_hi.ndim == 0:
            raw_hi = np.full((self._single_action_dim,), raw_hi.item(), dtype=np.float32)

        lo = self.process["action"].transform(raw_lo.reshape(1, -1))[0]
        hi = self.process["action"].transform(raw_hi.reshape(1, -1))[0]

        # The action encoder receives action chunks, so repeat the per-action
        # bound for every action in a chunk.
        lo = np.tile(lo, self._action_block)
        hi = np.tile(hi, self._action_block)
        return torch.as_tensor(lo, device=self.device, dtype=self.dtype), torch.as_tensor(
            hi, device=self.device, dtype=self.dtype
        )

    def _initial_action(self, batch_size: int, init_action: torch.Tensor | None):
        """Initialize exactly like the explicit diagram: one random sequence per env."""
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

            # WorldModelPolicy warm-starts with the remaining horizon. Pad if
            # necessary so the explicit rollout always sees the configured horizon.
            if actions.ndim == 4:
                # Defensive support for an old solver-style sample dimension.
                actions = actions[:, 0]

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
    # Explicit latent objective
    # ---------------------------------------------------------------------
    def _latent_energy(
        self,
        context_data: dict,
        goal_data: dict,
        act_seq: torch.Tensor,
        return_per_env: bool = False,
    ):
        # Encode context and goal without tracking gradients for the encoders,
        # exactly as in the supplied diagram.
        with torch.no_grad():
            ctx_output = self.model.encode(context_data)
            ctx_emb = ctx_output["emb"]
            ctx_act = ctx_output["act_emb"]

            goal_output = self.model.encode(goal_data)
            goal_emb = goal_output["emb"][:, -1]

        # Keep gradient tracking ONLY on the path from actions -> action
        # embedding -> model predictions -> final latent -> energy.
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

        # EXACT objective from the diagram:
        # energy = (pred - target)^2 summed over latent dimensions and batch,
        # implemented as a mean MSE scalar.
        sq_error = (final_pred_emb - goal_emb.detach()).pow(2)
        energy = sq_error.mean()

        if return_per_env:
            # Logging metric only; optimization still uses the exact global
            # mean scalar above, matching the supplied diagram.
            per_env_mse = sq_error.flatten(start_dim=1).mean(dim=1)
            return energy, per_env_mse.detach()

        return energy

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        if not self._configured:
            raise RuntimeError("ExplicitGradientSolver.configure() must be called before solve().")

        start_time = time.time()

        first = self._first_tensor(info_dict)
        batch_size = len(first)

        context = self._build_context(info_dict)
        goal = self._build_goal(info_dict)

        actions = self._initial_action(batch_size, init_action)

        energy_history = []
        bounds = self._normalized_bounds()

        for step in range(self.n_steps):
            # Rebuild the graph every iteration, as in the diagram.
            energy = self._latent_energy(context, goal, actions)

            # Explicit exact gradient with respect to action variables.
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
                        grad_energy = grad_energy * (
                            self.grad_clip / (grad_norm + 1e-6)
                        )

                # Manual gradient descent, exactly as in the supplied code.
                actions -= self.lr * grad_energy

                # Optional Langevin/action noise. With action_noise=0.0 this
                # contributes nothing, so the optimization is deterministic
                # after its initial torch.randn() action initialization.
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

        # Compute the final energy one more time so the returned cost is the
        # actual cost of the final optimized action sequence. Also log one
        # final latent MSE per episode/environment.
        final_energy, final_per_env_mse = self._latent_energy(
            context, goal, actions, return_per_env=True
        )
        energy_history.append(float(final_energy.detach().cpu().item()))

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

        print("\nFINAL LATENT MSE AFTER OPTIMIZATION")
        print("----------------------------------")
        for env_i, mse in enumerate(final_per_env_mse.cpu().tolist()):
            ep_label = (
                episode_ids[env_i]
                if episode_ids is not None and env_i < len(episode_ids)
                else env_i
            )
            print(f"Episode {ep_label}: final_MSE={mse:.8f}")
        print(f"Batch mean final MSE: {final_energy.item():.8f}")

        # WorldModelPolicy expects one plan per env. We return the action
        # sequence directly on CPU, without a sample dimension.
        actions_out = actions.detach().cpu()

        elapsed = time.time() - start_time
        print(
            f"ExplicitGradientSolver.solve completed in {elapsed:.4f}s "
            f"(final latent MSE={energy_history[-1]:.6f})."
        )

        return {
            "actions": actions_out,
            "cost": energy_history,
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

    # Create world environment.
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

        # Read the same gradient settings from the existing solver config.
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
    # Select valid evaluation starting points.
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
    # Evaluate with the normal stable-worldmodel environment loop.
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
            "planner: ExplicitGradientSolver; "
            "objective: mean((final_pred_emb - goal_emb)^2); "
            "gradient: torch.autograd.grad\n"
        )


if __name__ == "__main__":
    run()
