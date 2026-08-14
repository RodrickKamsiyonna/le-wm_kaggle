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
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy


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
# ManualGradSolver
#
# Replaces stable_worldmodel.solver.gd.GradientSolver entirely. No
# torch.optim object of any kind is used — every action update is
# `act_seq.data -= lr * act_seq.grad`, computed via a plain .backward() call.
# The rollout mechanics (encode -> iterative model.predict over the horizon
# -> MSE-to-goal loss) are copied straight from diagram.py's
# run_optimization_and_plot(), just batched over n_envs instead of a single
# instance, and the loss is (pred - goal).pow(2).sum(dim=-1) — NOT .mean().
#
# It still plugs into WorldModelPolicy (implements the `Solver` protocol:
# configure/.horizon/.action_dim/.n_envs/solve()) so the receding-horizon,
# warm-start, and per-env replanning bookkeeping isn't reimplemented here.
# ---------------------------------------------------------------------------
class ManualGradSolver:
    """Hand-written gradient-descent action solver (no stable_worldmodel solver classes)."""

    def __init__(
        self,
        model,
        process,
        transform,
        n_steps=50,
        lr=0.05,
        grad_clip=None,
        action_bounds=None,
        action_noise=0.0,
        device="cuda",
    ):
        self.model = model
        self.process = process
        self.transform = transform
        self.n_steps = n_steps
        self.lr = lr
        self.grad_clip = grad_clip
        self.action_bounds = action_bounds
        self.action_noise = action_noise
        self.device = torch.device(device)

        try:
            self.dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self.dtype = torch.float32

        # context window lookup
        self.history_size = getattr(model, "history_size", 1)
        self._configured = False

    def configure(self, *, action_space, n_envs, config):
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._raw_action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._raw_action_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def _prepare_frame_tensor(self, frame, transform_fn):
        t_frame = transform_fn(frame)
        if not isinstance(t_frame, torch.Tensor):
            t_frame = torch.as_tensor(t_frame)
        # Guarantee 3D shape: (C, H, W)
        while t_frame.dim() > 3:
            t_frame = t_frame.squeeze(0)
        return t_frame

    def _encode_context(self, pixels):
        """pixels: (n_envs, H, W, C) single current frame -> encoded context window."""
        t_fn = self.transform["pixels"]
        frames = torch.stack(
            [self._prepare_frame_tensor(p, t_fn) for p in pixels], dim=0
        ).to(self.device, dtype=self.dtype)  # (n_envs, C, H, W)

        # warm-start context window by repeating the current frame: (n_envs, history_size, C, H, W)
        batch = {"pixels": frames.unsqueeze(1).repeat(1, self.history_size, 1, 1, 1)}

        num_envs = pixels.shape[0] if hasattr(pixels, "shape") else len(pixels)
        if "action" in self.process:
            act_dim = self.process["action"].mean_.shape[0] * self._config.action_block
            batch["action"] = torch.zeros(
                num_envs,
                self.history_size,
                act_dim,
                device=self.device,
                dtype=self.dtype,
            )

        with torch.no_grad():
            out = self.model.encode(batch)
        return out["emb"], out["act_emb"]  # (n_envs, history_size, hidden_dim), (n_envs, history_size, embed_dim)

    def _encode_goal(self, goal_pixels):
        t_fn = self.transform["goal"]
        frames = torch.stack(
            [self._prepare_frame_tensor(p, t_fn) for p in goal_pixels], dim=0
        ).to(self.device, dtype=self.dtype)  # (n_envs, C, H, W)

        batch = {"pixels": frames.unsqueeze(1)}  # (n_envs, 1, C, H, W)
        num_envs = goal_pixels.shape[0] if hasattr(goal_pixels, "shape") else len(goal_pixels)
        if "action" in self.process:
            act_dim = self.process["action"].mean_.shape[0] * self._config.action_block
            batch["action"] = torch.zeros(
                num_envs, 1, act_dim, device=self.device, dtype=self.dtype
            )
        with torch.no_grad():
            out = self.model.encode(batch)
        return out["emb"][:, -1]  # (n_envs, hidden_dim)

    def solve(self, info_dict, init_action=None):
        n_envs = len(info_dict["pixels"])

        ctx_emb, ctx_act = self._encode_context(info_dict["pixels"])
        goal_emb = self._encode_goal(info_dict["goal"])

        # --- init action sequence, padding a shorter warm-started plan up to horizon ---
        if init_action is not None:
            init_action = init_action.to(self.device, dtype=self.dtype)
            remaining = self.horizon - init_action.shape[1]
            if remaining > 0:
                pad = torch.zeros(
                    n_envs, remaining, self.action_dim, device=self.device, dtype=self.dtype
                )
                init_action = torch.cat([init_action, pad], dim=1)
            act_seq = init_action.clone().detach()
        else:
            act_seq = torch.randn(
                n_envs, self.horizon, self.action_dim, device=self.device, dtype=self.dtype
            )

        if self.action_bounds and "action" in self.process:
            raw_lo, raw_hi = self.action_bounds
            proc = self.process["action"]
            norm_lo = proc.transform(np.array([[raw_lo] * self._raw_action_dim]))[0, 0]
            norm_hi = proc.transform(np.array([[raw_hi] * self._raw_action_dim]))[0, 0]
            act_seq.clamp_(norm_lo, norm_hi)

        act_seq.requires_grad_(True)
        cost_history = []

        # --- Raw gradient descent: cost.backward() + act_seq.grad, no torch.optim ---
        for step in range(self.n_steps):
            if act_seq.grad is not None:
                act_seq.grad = None

            act_emb_seq = self.model.action_encoder(act_seq)  # (n_envs, horizon, embed_dim)

            current_ctx_emb, current_ctx_act = ctx_emb, ctx_act
            final_pred_emb = None

            for t in range(self.horizon):
                step_act_emb = act_emb_seq[:, t : t + 1]
                full_act_ctx = torch.cat([current_ctx_act[:, 1:], step_act_emb], dim=1)
                pred_out = self.model.predict(current_ctx_emb, full_act_ctx)
                pred_emb = pred_out[:, -1] if pred_out.dim() == 3 else pred_out
                current_ctx_emb = torch.cat([current_ctx_emb[:, 1:], pred_emb.unsqueeze(1)], dim=1)
                current_ctx_act = full_act_ctx
                final_pred_emb = pred_emb

            # sum over latent dim, NOT mean — per-env MSE, shape (n_envs,)
            mse_loss = (final_pred_emb - goal_emb).pow(2).sum(dim=-1)
            cost = mse_loss.sum()  # sum across envs to get one scalar to backprop
            cost.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([act_seq], self.grad_clip)

            with torch.no_grad():
                act_seq -= self.lr * act_seq.grad
                if self.action_noise > 0.0:
                    act_seq += self.action_noise * torch.randn_like(act_seq)
                if self.action_bounds and "action" in self.process:
                    act_seq.clamp_(norm_lo, norm_hi)

            cost_history.append(mse_loss.detach().cpu())

            if (step + 1) % 10 == 0 or step == 0:
                print(f"    [solve] step {step+1:3d}/{self.n_steps}  mean MSE: {mse_loss.mean().item():.6f}")

        return {
            "actions": act_seq.detach().cpu(),
            "cost": torch.stack(cost_history, dim=1),  # (n_envs, n_steps)
        }


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
        ckpt_path = cfg.policy
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path += ".ckpt"

        print(f"Loading local PyTorch model from {ckpt_path}...")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg.plan_config)

        grad_cfg = cfg.get("gradient_solver", {})
        solver = ManualGradSolver(
            model=model,
            process=process,
            transform=transform,
            n_steps=grad_cfg.get("n_iter", 50),
            lr=grad_cfg.get("lr", 0.05),
            grad_clip=grad_cfg.get("grad_clip"),
            action_bounds=grad_cfg.get("action_bounds"),
            action_noise=grad_cfg.get("action_noise", 0.0),
            device="cuda",
        )

        policy = WorldModelPolicy(
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
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
