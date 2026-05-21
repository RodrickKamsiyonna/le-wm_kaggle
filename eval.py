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

    Treats the action sequence as a continuous variable and minimizes the
    world-model energy via gradient descent, consistent with the EQM training
    objective in lejepa_forward.

    The energy at each rollout step matches training exactly:
        E_t = (pred_emb_t - goal_emb).pow(2).sum(dim=-1).mean()
    i.e. sum over the hidden dimension, mean over the batch dimension.

    The predictor receives the full action-embedding context window at every
    step (matching how ARPredictor is called during training), not a single
    action slice.

    Args:
        model:          JEPA world model loaded via swm.wm.utils.load_pretrained.
                        Must expose .encode(), .predict(), .action_encoder().
        action_dim:     Dimensionality of a single action vector.
        horizon:        Number of rollout steps to optimise over.
        action_block:   How many of those steps to actually execute before
                        re-planning (receding-horizon control).
        ctx_len:        History window length used during training
                        (cfg.wm.history_size).  Needed to build the sliding
                        context correctly.
        n_iter:         Gradient-descent iterations per planning call.
        lr:             Adam learning rate.
        n_restarts:     Independent random restarts; best solution is kept.
        action_noise:   Std of Gaussian noise injected after every grad step
                        (helps escape shallow local minima).
        action_bounds:  Optional (low, high) tuple; actions are clamped after
                        every update.
    """

    def __init__(
        self,
        model,
        action_dim: int,
        horizon: int,
        action_block: int,
        ctx_len: int,
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
        self.ctx_len = ctx_len
        self.n_iter = n_iter
        self.lr = lr
        self.n_restarts = n_restarts
        self.action_noise = action_noise
        self.action_bounds = action_bounds

    @torch.no_grad()
    def _encode_context(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the current observation context.

        Returns:
            ctx_emb:  (1, ctx_len, hidden_dim)  — observation latents
            ctx_act:  (1, ctx_len, embed_dim)   — action embeddings for the
                      context window (used to seed the sliding window)
        """
        output = self.model.encode(batch)
        ctx_emb = output["emb"][:, : self.ctx_len]     # (1, ctx_len, hidden_dim)
        ctx_act = output["act_emb"][:, : self.ctx_len]  # (1, ctx_len, embed_dim)
        return ctx_emb, ctx_act

    def _compute_energy(self, ctx_emb, ctx_act, act_seq, goal_emb):
        # act_seq: (1, horizon, 2) → need (1, horizon, 10) for action_encoder
        # Build sliding windows of 5 actions, tiling at the boundaries
        B, H, D = act_seq.shape
        chunk = 5
        # Pad with the first action repeated at the front
        pad = act_seq[:, :1].expand(B, chunk - 1, D)   # (1, 4, 2)
        padded = torch.cat([pad, act_seq], dim=1)        # (1, horizon+4, 2)
        # Build (1, horizon, 10) by taking windows of 5
        windows = torch.stack(
            [padded[:, t:t+chunk].reshape(B, chunk*D) for t in range(H)], dim=1
        )  # (1, horizon, 10)
        
        # We need gradients to flow through the action_encoder
        act_emb_seq = self.model.action_encoder(windows)

        # Detach the initial context so we don't backprop into the visual encoder
        current_ctx_emb = ctx_emb.detach()   # (1, ctx_len, hidden_dim)
        current_ctx_act = ctx_act.detach()   # (1, ctx_len, embed_dim)

        total_energy = torch.zeros(1, device=ctx_emb.device, dtype=ctx_emb.dtype)

        for t in range(self.horizon):
            # ── build full-window action context for this step ──────────────
            # Append the candidate action at step t and drop the oldest entry,
            # giving a window of exactly ctx_len embeddings — matching training.
            step_act_emb = act_emb_seq[:, t : t + 1]              # (1,1,embed_dim)
            full_act_ctx = torch.cat(
                [current_ctx_act[:, 1:], step_act_emb], dim=1
            )  # (1, ctx_len, embed_dim)

            # ── predict next latent ─────────────────────────────────────────
            # predict(ctx_emb, ctx_act) → (1, ctx_len, hidden_dim) or
            # (1, hidden_dim); we take the last frame as the prediction.
            pred_out = self.model.predict(current_ctx_emb, full_act_ctx)
            if pred_out.dim() == 3:
                pred_emb = pred_out[:, -1]   # (1, hidden_dim)
            else:
                pred_emb = pred_out          # (1, hidden_dim)

            # ── energy: sum over hidden dim, mean over batch ────────────────
            # Accumulate energy across the entire rollout (dense reward mapping).
            total_energy = total_energy + (
                (pred_emb - goal_emb).pow(2).sum(dim=-1).mean()
            )

            # ── slide the observation context window ────────────────────────
            # CRITICAL FIX: DO NOT detach pred_emb here! 
            # Detaching here breaks Backpropagation Through Time (BPTT). We need 
            # the gradients from future steps to flow backwards through past predictions.
            current_ctx_emb = torch.cat(
                [current_ctx_emb[:, 1:], pred_emb.unsqueeze(1)], dim=1
            )
            current_ctx_act = full_act_ctx

        return total_energy

    def solve(
        self,
        batch: dict,
        goal_emb: torch.Tensor,  # (1, hidden_dim)
    ) -> torch.Tensor:
        """
        Return the best action block found across `n_restarts` initializations.

        Returns:
            Tensor of shape (action_block, action_dim) on CPU.
        """
        device = goal_emb.device
        dtype = goal_emb.dtype

        # goal_emb is squeezed to (1, hidden_dim) once here so _compute_energy
        # never has to handle variable leading dimensions.
        goal_emb = goal_emb.view(1, -1)   # (1, hidden_dim)

        ctx_emb, ctx_act = self._encode_context(batch)

        best_energy = float("inf")
        best_actions = None

        for _ in range(self.n_restarts):
            act_seq = torch.randn(
                1, self.horizon, self.action_dim, device=device, dtype=dtype
            )
            if self.action_bounds is not None:
                lo, hi = self.action_bounds
                act_seq.clamp_(lo, hi)
            act_seq = act_seq.requires_grad_(True)

            optimizer = torch.optim.Adam([act_seq], lr=self.lr)

            for _ in range(self.n_iter):
                optimizer.zero_grad()
                energy = self._compute_energy(ctx_emb, ctx_act, act_seq, goal_emb)
                energy.backward()
                optimizer.step()

                with torch.no_grad():
                    if self.action_noise > 0.0:
                        act_seq.data += self.action_noise * torch.randn_like(act_seq)
                    if self.action_bounds is not None:
                        lo, hi = self.action_bounds
                        act_seq.data.clamp_(lo, hi)

            # Evaluate final energy without building a new graph
            with torch.no_grad():
                final_energy = self._compute_energy(
                    ctx_emb, ctx_act, act_seq, goal_emb
                ).item()

            if final_energy < best_energy:
                best_energy = final_energy
                best_actions = act_seq.detach().clone()   # (1, horizon, action_dim)

        return best_actions[0, : self.action_block]   # (action_block, action_dim)


class GradientWorldModelPolicy:
    """
    Wraps GradientBasedSolver in the same interface as WorldModelPolicy so it
    can be passed to world.set_policy() unchanged.
    """

    def __init__(
        self,
        solver: GradientBasedSolver,
        config,           # swm.PlanConfig
        process: dict,    # sklearn scalers for normalisation
        transform: dict,  # image transforms
    ):
        self.solver = solver
        self.config = config
        self.process = process
        self.transform = transform
        
        self.env = None
        self.action_buffer = None
        self.steps_in_buffer = 0

    def set_env(self, env):
        """Satisfy the stable_worldmodel policy interface by linking the environment."""
        self.env = env

    def reset(self):
        """Clear the action buffer when the environment resets."""
        self.action_buffer = None
        self.steps_in_buffer = 0

    def _preprocess_obs(self, obs: dict) -> dict:
        batch = {}
        for key, val in obs.items():
            if key in self.transform:
                # Convert to numpy array safely just in case the env returned a tensor
                if torch.is_tensor(val):
                    val_np = val.detach().cpu().numpy()
                else:
                    val_np = np.array(val)
                
                # Check if the environment gave us a sequence of frames: (T, H, W, C)
                if val_np.ndim == 4:
                    # Apply the transform to each frame individually
                    frames = [self.transform[key](val_np[t]) for t in range(val_np.shape[0])]
                    # Stack into (T, C, H, W) and add the batch dimension -> (1, T, C, H, W)
                    batch[key] = torch.stack(frames, dim=0).unsqueeze(0)
                else:
                    # If it's a single image (H, W, C), transform and add batch & time dims -> (1, 1, C, H, W)
                    batch[key] = self.transform[key](val_np).unsqueeze(0).unsqueeze(0)
                    
            elif key in self.process:
                arr = np.array(val, dtype=np.float32).reshape(1, -1)
                transformed = self.process[key].transform(arr)  # (1, 2)
                t = torch.tensor(transformed, dtype=torch.float32)
            
                if key == "action":
                    # Model action_encoder expects (batch, seq_len, 10):
                    # 10 = 5 consecutive 2-dim actions flattened.
                    # At encode time we only have 1 action, so tile it 5x to fill the window.
                    tiled = t.repeat(1, 5)          # (1, 10)
                    batch[key] = tiled.unsqueeze(1) # (1, 1, 10)
                else:
                    batch[key] = t.unsqueeze(1)     # (1, 1, dim)
            else:
                try:
                    # Safely try to convert to float32 tensor
                    batch[key] = torch.tensor(
                        np.array(val, dtype=np.float32), dtype=torch.float32
                    ).unsqueeze(0)
                except (TypeError, ValueError):
                    # Skip metadata strings like 'env_name'
                    continue
        return batch
    
    @torch.no_grad()
    def _encode_goal(self, goal_obs: dict) -> torch.Tensor:
        """Encode goal observation → (1, hidden_dim)."""
        batch = self._preprocess_obs(goal_obs)
        device = next(self.solver.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        output = self.solver.model.encode(batch)
        return output["emb"][:, -1]   # (1, hidden_dim)

    def act(self, obs: dict, goal_obs: dict) -> np.ndarray:
        """
        Return the next action_block actions as (action_block, action_dim).
        """
        device = next(self.solver.model.parameters()).device

        batch = self._preprocess_obs(obs)
        batch = {k: v.to(device) for k, v in batch.items()}

        goal_emb = self._encode_goal(goal_obs)   # (1, hidden_dim)

        actions = self.solver.solve(batch, goal_emb)   # (action_block, action_dim)

        actions_np = actions.cpu().numpy()
        if "action" in self.process:
            actions_np = self.process["action"].inverse_transform(actions_np)

        return actions_np

    def get_action(self, infos: dict) -> np.ndarray:
        """
        Main entry point called by stable_worldmodel.World during evaluation.
        Handles batched environments and receding horizon action caching.
        """
        # Determine the number of parallel environments (batch size)
        num_envs = len(infos.get("pixels", infos.get("observation", [0])))

        # Replan only if buffer is empty or action block is exhausted
        if self.action_buffer is None or self.steps_in_buffer >= self.config.action_block:
            
            # Split 'infos' into current obs and goal obs
            obs_batch = {}
            goal_batch = {}
            for k, v in infos.items():
                if k == "goal":
                    goal_batch["pixels"] = v
                elif k.startswith("goal_"):
                    goal_batch[k.replace("goal_", "")] = v
                else:
                    obs_batch[k] = v

            planned_actions = []
            
            # Since GradientBasedSolver is hardcoded for batch_size=1, 
            # we loop over all environments to plan for them independently.
            for i in range(num_envs):
                single_obs = {k: v[i] for k, v in obs_batch.items() if v is not None}
                single_goal = {k: v[i] for k, v in goal_batch.items() if v is not None}
                
                # Returns (action_block, action_dim)
                acts = self.act(single_obs, single_goal)
                planned_actions.append(acts)
            
            # Stack into (num_envs, action_block, action_dim)
            self.action_buffer = np.stack(planned_actions, axis=0)
            self.steps_in_buffer = 0
            
        # Pop the next action for all environments: shape (num_envs, action_dim)
        actions = self.action_buffer[:, self.steps_in_buffer, :]
        self.steps_in_buffer += 1
        
        return actions

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation using gradient-based action planning."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"


    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    
    # Extract clean parameters that stable_worldmodel.World explicitly requires
    world = swm.World(
        env_name=cfg.world.env_name,
        num_envs=cfg.world.num_envs,
        max_episode_steps=cfg.world.max_episode_steps,
        image_shape=(224, 224)
    )

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
        ckpt_path = cfg.policy + "_object.ckpt"
        print(f"Loading direct model object from {ckpt_path}...")
        
        # 1. Bypass load_pretrained completely. Your checkpoint is a full 
        # PyTorch object, so we just load it directly.
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg.plan_config)
        grad_cfg = cfg.get("gradient_solver", {})
        
        # 2. Safely deduce the parameters that were missing from cfg.wm
        raw_action_dim = process["action"].mean_.shape[0]  # 2
        action_chunk = 5  # Conv1d has 10 input channels / 2 = 5
        action_dim = raw_action_dim  # solver still optimizes 2-dim actions per step

        # Determine history size (check eval config, model attributes, or default to 1)
        if hasattr(cfg, "wm") and hasattr(cfg.wm, "history_size"):
            ctx_len = cfg.wm.history_size
        elif hasattr(cfg, "model") and hasattr(cfg.model, "history_size"):
            ctx_len = cfg.model.history_size
        else:
            ctx_len = getattr(model, "history_size", 1)

        # 3. Initialize the solver
        solver = GradientBasedSolver(
            model=model,
            action_dim=action_dim,
            horizon=cfg.plan_config.horizon,
            action_block=cfg.plan_config.action_block,
            ctx_len=ctx_len,
            n_iter=grad_cfg.get("n_iter", 50),
            lr=grad_cfg.get("lr", 0.05),
            n_restarts=grad_cfg.get("n_restarts", 4),
            action_noise=grad_cfg.get("action_noise", 0.0),
            action_bounds=(
                tuple(grad_cfg.action_bounds)
                if grad_cfg.get("action_bounds")
                else None
            ),
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
