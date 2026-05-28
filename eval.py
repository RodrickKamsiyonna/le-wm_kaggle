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

    Optimizes over macro-actions (chunks) directly, matching the exact
    temporal stride and input dimensionality seen by the action_encoder
    during training.  Each optimized variable is a (action_chunk * action_dim)
    vector — identical in shape and semantics to one training sample row.

    To extract executable single-step actions we take only the LAST
    (most-recent) action_dim slice of the first chunk.  This is consistent
    with how the dataset is constructed: position [-action_dim:] of a chunk
    is the action that was actually executed at the corresponding time step.

    The energy is computed only at the terminal step to avoid penalising
    exploratory intermediate latent states:
        E = (final_pred_emb - goal_emb).pow(2).sum(dim=-1)

    Args:
        model:          JEPA world model loaded via swm.wm.utils.load_pretrained.
        action_dim:     Dimensionality of a single raw action vector (e.g. 2).
        horizon:        Number of latent rollout steps to optimise over.
        action_block:   How many raw single-step actions to execute before
                        re-planning (must be <= horizon).
        ctx_len:        History window length used during training
                        (cfg.wm.history_size).
        action_chunk:   Number of consecutive actions flattened into one
                        action-encoder input (== training frameskip).
        n_iter:         Gradient-descent iterations per planning call.
        lr:             Adam learning rate.
        grad_clip:      Max norm for gradient clipping on act_seq.
        n_restarts:     Independent random restarts; best solution is kept.
        action_noise:   Std of Gaussian noise injected after every grad step.
        action_bounds:  Optional (low, high) tuple in *normalised* space;
                        chunks are clamped element-wise after every update.
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
        assert action_block <= horizon, (
            f"action_block ({action_block}) must be <= horizon ({horizon})"
        )
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_block = action_block
        self.ctx_len = ctx_len
        self.action_chunk = action_chunk
        self.chunk_dim = action_chunk * action_dim  # full encoder input width
        self.n_iter = n_iter
        self.lr = lr
        self.grad_clip = grad_clip
        self.n_restarts = n_restarts
        self.action_noise = action_noise
        self.action_bounds = action_bounds

    @torch.no_grad()
    def _encode_context(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the current observation context into (ctx_emb, ctx_act)."""
        output = self.model.encode(batch)

        raw_emb = output["emb"]     # (1, T_obs, hidden_dim)
        raw_act = output["act_emb"] # (1, T_obs, embed_dim)

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

    def _compute_energy(
        self,
        ctx_emb: torch.Tensor,
        ctx_act: torch.Tensor,
        act_seq: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Roll out the world model for `horizon` steps and return the terminal
        squared-distance energy to goal_emb.

        act_seq: (1, horizon, chunk_dim)  — optimised macro-action sequence,
                 already in the same space as training rows (normalised,
                 chunk_dim = action_chunk * action_dim).
        """
        # Encode all horizon macro-actions in one batched call.
        # act_seq shape: (1, horizon, chunk_dim)
        act_emb_seq = self.model.action_encoder(act_seq)  # (1, horizon, embed_dim)

        current_ctx_emb = ctx_emb.detach()
        current_ctx_act = ctx_act.detach()

        final_pred_emb = None

        for t in range(self.horizon):
            step_act_emb = act_emb_seq[:, t : t + 1]           # (1, 1, embed_dim)
            full_act_ctx = torch.cat(
                [current_ctx_act[:, 1:], step_act_emb], dim=1  # slide window
            )

            pred_out = self.model.predict(current_ctx_emb, full_act_ctx)
            pred_emb = pred_out[:, -1] if pred_out.dim() == 3 else pred_out

            current_ctx_emb = torch.cat(
                [current_ctx_emb[:, 1:], pred_emb.unsqueeze(1)], dim=1
            )
            current_ctx_act = full_act_ctx

            final_pred_emb = pred_emb

        assert final_pred_emb is not None, "horizon must be >= 1"

        return (final_pred_emb - goal_emb).pow(2).sum(dim=-1)  # scalar per batch

    def solve(
        self,
        batch: dict,
        goal_emb: torch.Tensor,
    ) -> np.ndarray:
        """
        Return `action_block` normalised single-step actions as a numpy array
        of shape (action_block, action_dim).

        How single-step actions are extracted from optimised chunks
        ─────────────────────────────────────────────────────────────
        During training each dataset row contains a chunk of `action_chunk`
        consecutive normalised actions laid out as:

            [a_{t-k+1}, ..., a_{t-1}, a_t]   (oldest → newest)

        The action that was *executed* at step t is a_t, i.e. the last
        action_dim values of the chunk vector.  We therefore extract executed
        actions from the optimised chunks by taking the last action_dim values
        of each chunk:

            executed_action_i = act_seq[0, i, -action_dim:]

        This gives `horizon` normalised single-step actions.  We return the
        first `action_block` of them.
        """
        device = goal_emb.device
        dtype = goal_emb.dtype

        goal_emb = goal_emb.view(1, -1)
        ctx_emb, ctx_act = self._encode_context(batch)

        best_energy = float("inf")
        best_act_seq = None

        for _ in range(self.n_restarts):
            # Initialise in the same space as training rows.
            act_seq = torch.randn(
                1, self.horizon, self.chunk_dim, device=device, dtype=dtype
            )
            if self.action_bounds is not None:
                lo, hi = self.action_bounds
                act_seq.data.clamp_(lo, hi)
            act_seq = act_seq.requires_grad_(True)

            optimizer = torch.optim.Adam([act_seq], lr=self.lr)

            for _ in range(self.n_iter):
                optimizer.zero_grad()
                energy = self._compute_energy(ctx_emb, ctx_act, act_seq, goal_emb)
                energy.backward()

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
                final_energy = self._compute_energy(
                    ctx_emb, ctx_act, act_seq, goal_emb
                ).item()

            if final_energy < best_energy:
                best_energy = final_energy
                best_act_seq = act_seq.detach().clone()  # (1, horizon, chunk_dim)

        # ── Extract single-step actions from the optimised chunks ────────────
        # best_act_seq: (1, horizon, chunk_dim)
        # The last action_dim values of each chunk are the "executed" action
        # for that latent step, consistent with dataset construction.
        #
        # Result: (horizon, action_dim) normalised actions.
        executed = best_act_seq[0, :, -self.action_dim:]  # (horizon, action_dim)

        # Return the first action_block steps as a plain numpy array.
        # inverse_transform is applied by the caller (GradientWorldModelPolicy.act).
        return executed[: self.action_block].cpu().numpy()


class GradientWorldModelPolicy:
    """Wraps GradientBasedSolver in the same interface as WorldModelPolicy."""

    def __init__(
        self,
        solver: GradientBasedSolver,
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

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Convert a raw observation dict into a model-ready batch dict.

        For the action key we tile the single normalised action into a full
        chunk so the model's encode() receives the expected (chunk_dim,) input.
        All other non-image keys are normalised with their StandardScaler and
        left as single-timestep tensors.
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
                # StandardScaler operates on (N, action_dim) rows.
                transformed = self.process[key].transform(arr)  # (1, action_dim)
                t = torch.tensor(transformed, dtype=torch.float32)

                if key == "action":
                    # Tile into a fake chunk: [a, a, ..., a] of length chunk.
                    # This is only used for encoding the *context* observation,
                    # not for the planned actions, so the approximation is fine.
                    tiled = t.repeat(1, chunk)           # (1, chunk * action_dim)
                    batch[key] = tiled.unsqueeze(1)      # (1, 1, chunk_dim)
                else:
                    batch[key] = t.unsqueeze(1)          # (1, 1, feature_dim)

            else:
                try:
                    batch[key] = torch.tensor(
                        torch.from_numpy(np.array(val, dtype=np.float32))
                    ).unsqueeze(0)
                except (TypeError, ValueError):
                    continue

        return batch

    @torch.no_grad()
    def _encode_goal(self, goal_obs: dict) -> torch.Tensor:
        """Encode the goal observation and return its last embedding vector."""
        batch = self._preprocess_obs(goal_obs)
        device = next(self.solver.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        output = self.solver.model.encode(batch)
        return output["emb"][:, -1]  # (1, hidden_dim)

    def act(self, obs: dict, goal_obs: dict) -> np.ndarray:
        """
        Plan and return `action_block` raw (un-normalised) actions as
        (action_block, action_dim) numpy array.
        """
        device = next(self.solver.model.parameters()).device

        batch = self._preprocess_obs(obs)
        batch = {k: v.to(device) for k, v in batch.items()}
        goal_emb = self._encode_goal(goal_obs)

        # solve() returns normalised actions: (action_block, action_dim) numpy
        actions_norm = self.solver.solve(batch, goal_emb)

        # Invert normalisation to get raw environment actions.
        if "action" in self.process:
            actions_raw = self.process["action"].inverse_transform(actions_norm)
        else:
            actions_raw = actions_norm

        return actions_raw  # (action_block, action_dim)

    def get_action(self, infos: dict) -> np.ndarray:
        """
        Called every env step.  Re-plans when the action buffer is exhausted.
        Returns (num_envs, action_dim) raw actions.
        """
        first_val = next(iter(infos.values()))
        num_envs = len(first_val) if hasattr(first_val, "__len__") else 1

        if self.action_buffer is None or self.steps_in_buffer >= self.config.action_block:
            obs_batch: dict = {}
            goal_batch: dict = {}
            for k, v in infos.items():
                if k == "goal":
                    goal_batch["pixels"] = v
                elif k.startswith("goal_"):
                    goal_batch[k[len("goal_"):]] = v
                else:
                    obs_batch[k] = v

            planned_actions = []
            for i in range(num_envs):
                single_obs  = {k: v[i] for k, v in obs_batch.items()  if v is not None}
                single_goal = {k: v[i] for k, v in goal_batch.items() if v is not None}
                acts = self.act(single_obs, single_goal)  # (action_block, action_dim)
                planned_actions.append(acts)

            self.action_buffer = np.stack(planned_actions, axis=0)  # (num_envs, action_block, action_dim)
            self.steps_in_buffer = 0

        actions = self.action_buffer[:, self.steps_in_buffer, :]  # (num_envs, action_dim)
        self.steps_in_buffer += 1
        return actions


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget

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

    # ── Build per-column StandardScalers ─────────────────────────────────────
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

        config  = swm.PlanConfig(**cfg.plan_config)
        grad_cfg = cfg.get("gradient_solver", {})

        # action_dim: dimensionality of a single raw action step
        raw_action_dim = process["action"].mean_.shape[0]

        # action_chunk: how many consecutive steps the encoder expects
        if hasattr(cfg, "wm") and hasattr(cfg.wm, "action_chunk"):
            action_chunk = cfg.wm.action_chunk
        elif hasattr(model, "action_encoder") and hasattr(
            model.action_encoder, "in_features"
        ):
            action_chunk = model.action_encoder.in_features // raw_action_dim
        else:
            action_chunk = grad_cfg.get("action_chunk", 5)

        # ctx_len: observation history window
        if hasattr(cfg, "wm") and hasattr(cfg.wm, "history_size"):
            ctx_len = cfg.wm.history_size
        elif hasattr(cfg, "model") and hasattr(cfg.model, "history_size"):
            ctx_len = cfg.model.history_size
        else:
            ctx_len = getattr(model, "history_size", 1)

        # ── Action bounds in normalised space ─────────────────────────────────
        # The StandardScaler is fit on single-step (action_dim,) rows, so we
        # transform a single-step bound vector and read off the scalar.
        # We apply it element-wise to the full chunk tensor, which is correct
        # as long as the scaler was fit on the same action space (it was).
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

        solver = GradientBasedSolver(
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
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes  = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

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

    out_path = results_path / cfg.output.filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
