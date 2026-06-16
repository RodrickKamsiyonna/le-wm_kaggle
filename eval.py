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


class BatchedGradientSolver:
    """
    Batched gradient-based action sequence optimizer.

    Identical to GradientBasedSolver but operates over a batch of B
    environments simultaneously.  All rollouts share the same model weights
    but have independent action sequences, so the single forward/backward
    pass scales linearly in B and avoids repeated Python-level loops.

    The energy tensor has shape (B,) and gradients flow independently to
    each environment's act_seq slice.

    Args:
        model:          JEPA world model.
        action_dim:     Dimensionality of a single raw action vector.
        horizon:        Number of latent rollout steps.
        action_block:   Steps to execute before re-planning.
        ctx_len:        History window length (cfg.wm.history_size).
        action_chunk:   Consecutive actions per encoder input.
        n_iter:         Gradient-descent iterations.
        lr:             Adam learning rate.
        grad_clip:      Max grad norm.
        n_restarts:     Independent random restarts per planning call.
        action_noise:   Std of Gaussian noise after each grad step.
        action_bounds:  Optional (low, high) clamp in normalised space.
    """

    def __init__(
        self,
        model,
        action_dim: int,
        horizon: int,
        action_block: int,
        ctx_len: int,
        action_chunk: int = 5,
        n_iter: int = 50,
        lr: float = 0.05,
        grad_clip: float = 1.0,
        n_restarts: int = 4,
        action_noise: float = 0.0,
        action_bounds: tuple | None = None,
    ):
        assert action_block <= horizon
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_block = action_block
        self.ctx_len = ctx_len
        self.action_chunk = action_chunk
        self.chunk_dim = action_chunk * action_dim
        self.n_iter = n_iter
        self.lr = lr
        self.grad_clip = grad_clip
        self.n_restarts = n_restarts
        self.action_noise = action_noise
        self.action_bounds = action_bounds

    @torch.no_grad()
    def _encode_context_batch(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the batch of observations.

        batch values have leading dim B (already stacked by the caller).
        Returns ctx_emb, ctx_act both of shape (B, ctx_len, D).
        """
        output = self.model.encode(batch)
        raw_emb = output["emb"]      # (B, T_obs, hidden_dim)
        raw_act = output["act_emb"]  # (B, T_obs, embed_dim)

        T_obs = raw_emb.shape[1]
        if T_obs >= self.ctx_len:
            ctx_emb = raw_emb[:, -self.ctx_len:]
            ctx_act = raw_act[:, -self.ctx_len:]
        else:
            pad_len = self.ctx_len - T_obs
            ctx_emb = torch.cat(
                [raw_emb[:, :1].expand(-1, pad_len, -1), raw_emb], dim=1
            )
            ctx_act = torch.cat(
                [raw_act[:, :1].expand(-1, pad_len, -1), raw_act], dim=1
            )
        return ctx_emb, ctx_act

    def _compute_energy_batch(
        self,
        ctx_emb: torch.Tensor,   # (B, ctx_len, hidden_dim)
        ctx_act: torch.Tensor,   # (B, ctx_len, embed_dim)
        act_seq: torch.Tensor,   # (B, horizon, chunk_dim)
        goal_emb: torch.Tensor,  # (B, hidden_dim)
    ) -> torch.Tensor:
        """Return per-env terminal squared-distance energy, shape (B,)."""
        # Encode all macro-actions in one batched call: (B, horizon, embed_dim)
        act_emb_seq = self.model.action_encoder(act_seq)

        current_ctx_emb = ctx_emb.detach()
        current_ctx_act = ctx_act.detach()

        final_pred_emb = None

        for t in range(self.horizon):
            step_act_emb = act_emb_seq[:, t : t + 1]           # (B, 1, embed_dim)
            full_act_ctx = torch.cat(
                [current_ctx_act[:, 1:], step_act_emb], dim=1
            )

            pred_out = self.model.predict(current_ctx_emb, full_act_ctx)
            pred_emb = pred_out[:, -1] if pred_out.dim() == 3 else pred_out  # (B, D)

            current_ctx_emb = torch.cat(
                [current_ctx_emb[:, 1:], pred_emb.unsqueeze(1)], dim=1
            )
            current_ctx_act = full_act_ctx
            final_pred_emb = pred_emb

        assert final_pred_emb is not None
        # (B,)
        return (final_pred_emb - goal_emb).pow(2).sum(dim=-1)

    def solve_batch(
        self,
        batch: dict,             # already stacked over B envs, on device
        goal_emb: torch.Tensor,  # (B, hidden_dim)
    ) -> np.ndarray:
        """
        Plan for all B envs simultaneously.

        Returns (B, action_block, action_dim) normalised numpy array.
        """
        B = goal_emb.shape[0]
        device = goal_emb.device
        dtype = goal_emb.dtype

        ctx_emb, ctx_act = self._encode_context_batch(batch)

        best_energy = torch.full((B,), float("inf"), device=device, dtype=dtype)
        best_act_seq = torch.zeros(
            B, self.horizon, self.chunk_dim, device=device, dtype=dtype
        )

        for _ in range(self.n_restarts):
            act_seq = torch.randn(
                B, self.horizon, self.chunk_dim, device=device, dtype=dtype
            )
            if self.action_bounds is not None:
                lo, hi = self.action_bounds
                act_seq.data.clamp_(lo, hi)
            act_seq = act_seq.requires_grad_(True)

            optimizer = torch.optim.Adam([act_seq], lr=self.lr)

            for _ in range(self.n_iter):
                optimizer.zero_grad()
                # energy shape: (B,); sum so gradients flow to all envs
                energy = self._compute_energy_batch(ctx_emb, ctx_act, act_seq, goal_emb)
                energy.sum().backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_([act_seq], self.grad_clip)

                optimizer.step()

                with torch.no_grad():
                    if self.action_noise > 0.0:
                        act_seq.data += self.action_noise * torch.randn_like(act_seq)
                    if self.action_bounds is not None:
                        lo, hi = self.action_bounds
                        act_seq.data.clamp_(lo, hi)

            with torch.no_grad():
                final_energy = self._compute_energy_batch(
                    ctx_emb, ctx_act, act_seq, goal_emb
                )  # (B,)

            # Per-env best tracking
            improved = final_energy < best_energy  # (B,) bool
            best_energy = torch.where(improved, final_energy, best_energy)
            best_act_seq[improved] = act_seq.detach()[improved]

        # Extract single-step actions: last action_dim values of each chunk
        # best_act_seq: (B, horizon, chunk_dim)
        executed = best_act_seq[:, :, -self.action_dim:]  # (B, horizon, action_dim)
        return executed[:, : self.action_block].cpu().numpy()  # (B, action_block, action_dim)


class BatchedGradientWorldModelPolicy:
    """
    Batched planner: solves all environments in a single forward+backward pass
    instead of iterating over them one by one.
    """

    def __init__(
        self,
        solver: BatchedGradientSolver,
        config,
        process: dict,
        transform: dict,
    ):
        self.solver = solver
        self.config = config
        self.process = process
        self.transform = transform

        self.env = None
        self.action_buffer = None
        self.steps_in_buffer = 0

    def set_env(self, env):
        self.env = env

    def reset(self):
        self.action_buffer = None
        self.steps_in_buffer = 0

    def _preprocess_single(self, obs: dict) -> dict:
        """
        Preprocess one environment's observation dict into tensors with a
        leading time-step dimension (1, T, ...).  Mirrors the original policy.
        """
        batch = {}
        chunk = self.solver.action_chunk

        for key, val in obs.items():
            if key in self.transform:
                val_np = (
                    val.detach().cpu().numpy() if torch.is_tensor(val) else np.array(val)
                )
                if val_np.ndim == 4:
                    frames = [self.transform[key](val_np[t]) for t in range(val_np.shape[0])]
                    batch[key] = torch.stack(frames, dim=0).unsqueeze(0)
                else:
                    batch[key] = self.transform[key](val_np).unsqueeze(0).unsqueeze(0)

            elif key in self.process:
                arr = np.array(val, dtype=np.float32).reshape(1, -1)
                transformed = self.process[key].transform(arr)
                t = torch.tensor(transformed, dtype=torch.float32)
                if key == "action":
                    tiled = t.repeat(1, chunk)
                    batch[key] = tiled.unsqueeze(1)
                else:
                    batch[key] = t.unsqueeze(1)
            else:
                try:
                    if torch.is_tensor(val):
                        batch[key] = val.detach().clone().to(dtype=torch.float32).unsqueeze(0)
                    elif isinstance(val, np.ndarray):
                        batch[key] = torch.from_numpy(val.astype(np.float32)).unsqueeze(0)
                    else:
                        batch[key] = torch.tensor(val, dtype=torch.float32).unsqueeze(0)
                except (TypeError, ValueError):
                    continue
        return batch

    def _stack_batch(self, list_of_dicts: list[dict]) -> dict:
        """
        Stack a list of per-env preprocessed dicts into a single batched dict.

        Each value in the individual dicts has shape (1, T, ...).
        After stacking the leading env dim: (B, T, ...).
        """
        keys = list_of_dicts[0].keys()
        return {
            k: torch.cat([d[k] for d in list_of_dicts], dim=0)
            for k in keys
            if all(k in d for d in list_of_dicts)
        }

    @torch.no_grad()
    def _encode_goal_batch(self, goal_obs_list: list[dict]) -> torch.Tensor:
        """Encode B goal observations and return (B, hidden_dim) embeddings."""
        device = next(self.solver.model.parameters()).device
        preprocessed = [self._preprocess_single(g) for g in goal_obs_list]
        batched = self._stack_batch(preprocessed)
        batched = {k: v.to(device) for k, v in batched.items()}
        output = self.solver.model.encode(batched)
        return output["emb"][:, -1]  # (B, hidden_dim)

    def act_batch(
        self,
        obs_list: list[dict],
        goal_obs_list: list[dict],
    ) -> np.ndarray:
        """
        Plan for all B environments at once.

        Returns (B, action_block, action_dim) raw (un-normalised) numpy array.
        """
        device = next(self.solver.model.parameters()).device

        preprocessed_obs = [self._preprocess_single(o) for o in obs_list]
        batched_obs = self._stack_batch(preprocessed_obs)
        batched_obs = {k: v.to(device) for k, v in batched_obs.items()}

        goal_emb = self._encode_goal_batch(goal_obs_list)  # (B, hidden_dim)

        # Batched solve: (B, action_block, action_dim) normalised
        actions_norm = self.solver.solve_batch(batched_obs, goal_emb)

        # Invert normalisation
        if "action" in self.process:
            B, T, D = actions_norm.shape
            # inverse_transform works on (N, action_dim) flat arrays
            flat = actions_norm.reshape(B * T, D)
            flat_raw = self.process["action"].inverse_transform(flat)
            actions_raw = flat_raw.reshape(B, T, D)
        else:
            actions_raw = actions_norm

        return actions_raw  # (B, action_block, action_dim)

    def get_action(self, infos: dict) -> np.ndarray:
        """
        Called every env step.  Re-plans the full batch when the buffer
        is exhausted.  Returns (num_envs, action_dim) raw actions.
        """
        first_val = next(iter(infos.values()))
        num_envs = len(first_val) if hasattr(first_val, "__len__") else 1

        if (
            self.action_buffer is None
            or self.steps_in_buffer >= self.config.action_block
        ):
            obs_list: list[dict] = []
            goal_list: list[dict] = []

            for i in range(num_envs):
                obs_i: dict = {}
                goal_i: dict = {}
                for k, v in infos.items():
                    if v is None:
                        continue
                    if k == "goal":
                        goal_i["pixels"] = v[i]
                    elif k.startswith("goal_"):
                        goal_i[k[len("goal_"):]] = v[i]
                    else:
                        obs_i[k] = v[i]
                obs_list.append(obs_i)
                goal_list.append(goal_i)

            # Single batched planning call for all envs
            planned = self.act_batch(obs_list, goal_list)
            # planned: (num_envs, action_block, action_dim)
            self.action_buffer = planned
            self.steps_in_buffer = 0

        actions = self.action_buffer[:, self.steps_in_buffer, :]  # (num_envs, action_dim)
        self.steps_in_buffer += 1
        return actions


def _extract_success_rate(metrics: dict) -> float:
    """
    Pull the scalar success rate out of whatever structure
    world.evaluate_from_dataset returns.

    Tries common key names in order; falls back to the first numeric value
    found if none match.
    """
    candidates = ("success_rate", "success", "avg_success", "mean_success")
    for key in candidates:
        if key in metrics:
            val = metrics[key]
            return float(val) if not hasattr(val, "__len__") else float(np.mean(val))

    # Fallback: first numeric scalar in the dict
    for val in metrics.values():
        try:
            return float(val)
        except (TypeError, ValueError):
            continue

    raise KeyError(
        f"Could not find a success-rate field in metrics dict. Keys: {list(metrics.keys())}"
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget

    # Number of independent experiment repetitions (override via
    # +num_experiments=N on the command line if needed).
    num_experiments: int = cfg.get("num_experiments", 10)

    world = swm.World(
        env_name=cfg.world.env_name,
        num_envs=cfg.world.num_envs,
        max_episode_steps=cfg.world.max_episode_steps,
        image_shape=(224, 224),
    )

    transform = {
        "pixels": img_transform(cfg),
        "goal":   img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    # ── Build per-column StandardScalers (once, shared across experiments) ───
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
        ckpt_path = cfg.policy + "_object.ckpt"
        print(f"Loading model object from {ckpt_path}...")

        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg.plan_config)
        grad_cfg = cfg.get("gradient_solver", {})

        raw_action_dim = process["action"].mean_.shape[0]

        if hasattr(cfg, "wm") and hasattr(cfg.wm, "action_chunk"):
            action_chunk = cfg.wm.action_chunk
        elif hasattr(model, "action_encoder") and hasattr(
            model.action_encoder, "in_features"
        ):
            action_chunk = model.action_encoder.in_features // raw_action_dim
        else:
            action_chunk = grad_cfg.get("action_chunk", 5)

        if hasattr(cfg, "wm") and hasattr(cfg.wm, "history_size"):
            ctx_len = cfg.wm.history_size
        elif hasattr(cfg, "model") and hasattr(cfg.model, "history_size"):
            ctx_len = cfg.model.history_size
        else:
            ctx_len = getattr(model, "history_size", 1)

        action_bounds = None
        if grad_cfg.get("action_bounds"):
            raw_lo, raw_hi = grad_cfg.action_bounds
            norm_lo = process["action"].transform(
                np.array([[raw_lo] * raw_action_dim], dtype=np.float32)
            )[0, 0]
            norm_hi = process["action"].transform(
                np.array([[raw_hi] * raw_action_dim], dtype=np.float32)
            )[0, 0]
            action_bounds = (norm_lo, norm_hi)

        solver = BatchedGradientSolver(
            model=model,
            action_dim=raw_action_dim,
            horizon=cfg.plan_config.horizon,
            action_block=cfg.plan_config.action_block,
            ctx_len=ctx_len,
            action_chunk=action_chunk,
            n_iter=grad_cfg.get("n_iter", 50),
            lr=grad_cfg.get("lr", 0.05),
            grad_clip=grad_cfg.get("grad_clip", 1.0),
            n_restarts=grad_cfg.get("n_restarts", 4),
            action_noise=grad_cfg.get("action_noise", 0.0),
            action_bounds=action_bounds,
        )

        policy = BatchedGradientWorldModelPolicy(
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
    results_path.mkdir(parents=True, exist_ok=True)

    # ── Pre-compute valid starting points once ───────────────────────────────
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask    = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    world.set_policy(policy)

    # ── Repeated experiment loop ─────────────────────────────────────────────
    # Each experiment draws a fresh independent random sample of episodes so
    # that the per-experiment success rates are not correlated by sample overlap.
    # We derive each experiment's seed deterministically from cfg.seed so that
    # runs are fully reproducible.
    all_success_rates: list[float] = []
    all_metrics: list[dict] = []
    total_start = time.time()

    print(f"\nRunning {num_experiments} independent experiments "
          f"({cfg.eval.num_eval} episodes each) …\n")

    for exp_idx in range(num_experiments):
        exp_seed = cfg.seed + exp_idx  # deterministic but distinct per experiment
        g = np.random.default_rng(exp_seed)

        random_episode_indices = g.choice(
            len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
        )
        random_episode_indices = np.sort(valid_indices[random_episode_indices])

        eval_episodes  = dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

        if len(eval_episodes) < cfg.eval.num_eval:
            raise ValueError(
                f"Experiment {exp_idx}: not enough episodes with sufficient length."
            )

        # Reset the policy buffer between experiments so stale actions from
        # the previous run don't bleed into the next one.
        if hasattr(policy, "reset"):
            policy.reset()

        exp_start = time.time()
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video_path=results_path,
        )
        exp_elapsed = time.time() - exp_start

        success_rate = _extract_success_rate(metrics)
        all_success_rates.append(success_rate)
        all_metrics.append(metrics)

        print(
            f"  Experiment {exp_idx + 1:3d}/{num_experiments}  "
            f"success_rate={success_rate:.4f}  "
            f"elapsed={exp_elapsed:.1f}s"
        )

    total_elapsed = time.time() - total_start

    # ── Aggregate statistics ─────────────────────────────────────────────────
    rates = np.array(all_success_rates)
    mean_sr  = float(np.mean(rates))
    std_sr   = float(np.std(rates, ddof=1))   # sample std (ddof=1)
    # 95 % confidence interval via t-distribution approximation (large n → ≈ 1.96)
    se_sr    = std_sr / np.sqrt(len(rates))
    ci95_lo  = mean_sr - 1.96 * se_sr
    ci95_hi  = mean_sr + 1.96 * se_sr

    summary = (
        f"\n{'=' * 60}\n"
        f"REPEATED EVALUATION SUMMARY  ({num_experiments} experiments)\n"
        f"{'=' * 60}\n"
        f"  Mean success rate : {mean_sr:.4f}\n"
        f"  Std  success rate : {std_sr:.4f}\n"
        f"  95% CI            : [{ci95_lo:.4f}, {ci95_hi:.4f}]\n"
        f"  Min / Max         : {rates.min():.4f} / {rates.max():.4f}\n"
        f"  Total time        : {total_elapsed:.1f}s\n"
        f"{'=' * 60}\n"
    )
    print(summary)

    # ── Persist results ──────────────────────────────────────────────────────
    out_path = results_path / cfg.output.filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write(summary)
        f.write("\n==== PER-EXPERIMENT RESULTS ====\n")
        for i, (sr, m) in enumerate(zip(all_success_rates, all_metrics)):
            f.write(f"experiment_{i:03d}: success_rate={sr:.4f}  metrics={m}\n")
        f.write(f"\ntotal_evaluation_time: {total_elapsed:.1f} seconds\n")


if __name__ == "__main__":
    run()
