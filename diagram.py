"""
mse_diagnostic.py
─────────────────
Pick one sample from the test set, run the gradient planner, and record the
MSE between the predicted terminal embedding and the goal embedding at every
optimisation iteration across all restarts.

Output: mse_over_time.png  (MSE on y-axis, iteration on x-axis)

Usage:
    python mse_diagnostic.py  [same hydra overrides as eval]
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig
from pathlib import Path
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


# ── Helpers copied from eval (keep self-contained) ───────────────────────────

def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


# ── MSE-recording solver ──────────────────────────────────────────────────────

class DiagnosticSolver:
    """
    Same optimisation logic as GradientBasedSolver / BatchedGradientSolver,
    but records the MSE to the goal embedding after every gradient step.

    The recorded curve spans all restarts concatenated:
        [restart_0_iter_0, ..., restart_0_iter_N,
         restart_1_iter_0, ..., restart_R_iter_N]

    Attributes
    ----------
    mse_history : list[float]
        MSE values appended in-place during solve().
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
        self.mse_history: list[float] = []

    @torch.no_grad()
    def _encode_context(self, batch):
        output = self.model.encode(batch)
        raw_emb = output["emb"]
        raw_act = output["act_emb"]
        T_obs = raw_emb.shape[1]
        if T_obs >= self.ctx_len:
            ctx_emb = raw_emb[:, -self.ctx_len:]
            ctx_act = raw_act[:, -self.ctx_len:]
        else:
            pad_len = self.ctx_len - T_obs
            ctx_emb = torch.cat([raw_emb[:, :1].expand(-1, pad_len, -1), raw_emb], dim=1)
            ctx_act = torch.cat([raw_act[:, :1].expand(-1, pad_len, -1), raw_act], dim=1)
        return ctx_emb, ctx_act

    def _rollout_terminal(self, ctx_emb, ctx_act, act_seq):
        """
        Roll out horizon steps and return the terminal predicted embedding.
        act_seq: (1, horizon, chunk_dim) — may or may not require grad.
        """
        act_emb_seq = self.model.action_encoder(act_seq)  # (1, horizon, embed_dim)

        cur_emb = ctx_emb.detach()
        cur_act = ctx_act.detach()
        final_emb = None

        for t in range(self.horizon):
            step_act = act_emb_seq[:, t : t + 1]
            full_act = torch.cat([cur_act[:, 1:], step_act], dim=1)
            pred_out = self.model.predict(cur_emb, full_act)
            pred_emb = pred_out[:, -1] if pred_out.dim() == 3 else pred_out
            cur_emb = torch.cat([cur_emb[:, 1:], pred_emb.unsqueeze(1)], dim=1)
            cur_act = full_act
            final_emb = pred_emb

        return final_emb  # (1, hidden_dim)

    def solve(self, batch: dict, goal_emb: torch.Tensor) -> np.ndarray:
        """
        Optimise and record MSE at each iteration.
        Returns (action_block, action_dim) normalised numpy array.
        """
        self.mse_history.clear()

        device = goal_emb.device
        dtype = goal_emb.dtype
        goal_emb = goal_emb.view(1, -1)

        ctx_emb, ctx_act = self._encode_context(batch)

        best_energy = float("inf")
        best_act_seq = None

        for restart in range(self.n_restarts):
            act_seq = torch.randn(
                1, self.horizon, self.chunk_dim, device=device, dtype=dtype
            )
            if self.action_bounds is not None:
                act_seq.data.clamp_(*self.action_bounds)
            act_seq = act_seq.requires_grad_(True)

            optimizer = torch.optim.Adam([act_seq], lr=self.lr)

            for step in range(self.n_iter):
                optimizer.zero_grad()
                terminal = self._rollout_terminal(ctx_emb, ctx_act, act_seq)
                energy = (terminal - goal_emb).pow(2).sum(dim=-1)  # scalar
                energy.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_([act_seq], self.grad_clip)

                optimizer.step()

                with torch.no_grad():
                    if self.action_noise > 0.0:
                        act_seq.data += self.action_noise * torch.randn_like(act_seq)
                    if self.action_bounds is not None:
                        act_seq.data.clamp_(*self.action_bounds)

                    # Record MSE (mean over hidden dim, not sum) for readability
                    t_emb = self._rollout_terminal(ctx_emb, ctx_act, act_seq)
                    mse = (t_emb - goal_emb).pow(2).mean().item()
                    self.mse_history.append(mse)

            with torch.no_grad():
                final_energy = (
                    (self._rollout_terminal(ctx_emb, ctx_act, act_seq) - goal_emb)
                    .pow(2)
                    .sum(dim=-1)
                    .item()
                )

            if final_energy < best_energy:
                best_energy = final_energy
                best_act_seq = act_seq.detach().clone()

        executed = best_act_seq[0, :, -self.action_dim:]
        return executed[: self.action_block].cpu().numpy()


# ── Preprocessing (mirrors eval.py) ─────────────────────────────────────────

def preprocess_obs(obs, process, transform, action_chunk):
    batch = {}
    for key, val in obs.items():
        if key in transform:
            val_np = val.detach().cpu().numpy() if torch.is_tensor(val) else np.array(val)
            if val_np.ndim == 4:
                frames = [transform[key](val_np[t]) for t in range(val_np.shape[0])]
                batch[key] = torch.stack(frames, dim=0).unsqueeze(0)
            else:
                batch[key] = transform[key](val_np).unsqueeze(0).unsqueeze(0)
        elif key in process:
            arr = np.array(val, dtype=np.float32).reshape(1, -1)
            t = torch.tensor(process[key].transform(arr), dtype=torch.float32)
            if key == "action":
                batch[key] = t.repeat(1, action_chunk).unsqueeze(1)
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


@torch.no_grad()
def encode_goal(obs_dict, model, process, transform, action_chunk, device):
    batch = preprocess_obs(obs_dict, process, transform, action_chunk)
    batch = {k: v.to(device) for k, v in batch.items()}
    return model.encode(batch)["emb"][:, -1]


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.get("policy", "random") != "random", (
        "mse_diagnostic.py requires a trained model; set policy= in your config."
    )

    ckpt_path = cfg.policy + "_object.ckpt"
    print(f"Loading model from {ckpt_path} ...")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    device = next(model.parameters()).device

    transform = {
        "pixels": img_transform(cfg),
        "goal":   img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    # Build scalers
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler

    grad_cfg = cfg.get("gradient_solver", {})
    raw_action_dim = process["action"].mean_.shape[0]

    if hasattr(cfg, "wm") and hasattr(cfg.wm, "action_chunk"):
        action_chunk = cfg.wm.action_chunk
    elif hasattr(model, "action_encoder") and hasattr(model.action_encoder, "in_features"):
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

    # ── Pick a single valid sample ────────────────────────────────────────────
    episode_len_arr = []
    step_idx_all = dataset.get_col_data("step_idx")
    ep_idx_all   = dataset.get_col_data(col_name)
    for ep_id in ep_indices:
        episode_len_arr.append(np.max(step_idx_all[ep_idx_all == ep_id]) + 1)
    episode_len_arr = np.array(episode_len_arr)

    max_start_idx = episode_len_arr - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row  = np.array(
        [max_start_idx_dict[ep_id] for ep_id in ep_idx_all]
    )
    valid_mask    = step_idx_all <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    rng = np.random.default_rng(cfg.seed)
    sample_row = valid_indices[rng.integers(len(valid_indices))]

    ep_id    = ep_idx_all[sample_row]
    start_st = step_idx_all[sample_row]
    goal_step_row = sample_row + cfg.eval.goal_offset_steps

    print(f"Diagnostic sample: episode {ep_id}, start step {start_st}")

    # ── Load obs and goal from dataset ────────────────────────────────────────
    obs_data  = dataset.get_row_data(np.array([sample_row]))
    goal_data = dataset.get_row_data(np.array([goal_step_row]))

    def row_to_obs(data):
        obs = {}
        for key in cfg.dataset.keys_to_cache:
            if key not in data:
                continue
            val = data[key]
            if hasattr(val, "__len__"):
                obs[key] = val[0]
            else:
                obs[key] = val
        return obs

    obs_dict  = row_to_obs(obs_data)
    goal_dict = row_to_obs(goal_data)

    # Rename goal pixel key if present
    if "pixels" in goal_dict:
        goal_dict_renamed = {"pixels": goal_dict["pixels"]}
    else:
        goal_dict_renamed = goal_dict

    # ── Encode goal ───────────────────────────────────────────────────────────
    goal_emb = encode_goal(
        goal_dict_renamed, model, process, transform, action_chunk, device
    )  # (1, hidden_dim)

    # ── Build diagnostic solver ───────────────────────────────────────────────
    solver = DiagnosticSolver(
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

    # ── Pre-process observation ───────────────────────────────────────────────
    batch = preprocess_obs(obs_dict, process, transform, action_chunk)
    batch = {k: v.to(device) for k, v in batch.items()}

    # ── Run optimisation (records MSE internally) ─────────────────────────────
    print("Running gradient optimisation and recording MSE ...")
    solver.solve(batch, goal_emb)

    mse_curve = np.array(solver.mse_history)
    print(f"Initial MSE: {mse_curve[0]:.6f}  →  Final MSE: {mse_curve[-1]:.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    n_iter    = grad_cfg.get("n_iter", 50)
    n_restart = grad_cfg.get("n_restarts", 4)

    # Shade restart boundaries
    for r in range(1, n_restart):
        ax.axvline(r * n_iter, color="#cccccc", linewidth=0.8, linestyle="--")

    ax.plot(mse_curve, color="#2563eb", linewidth=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = Path("mse_over_time.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    run()
