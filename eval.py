import os
os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import stable_worldmodel as swm


# ----------------------------
# Utils from your scripts
# ----------------------------
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


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


class SingleInstancePreprocessor:
    """
    Handles normalisation and transform for a single data instance
    to prepare it for the World Model.
    Exact same logic as diagram.py
    """
    def __init__(self, config, process: dict, transform: dict, action_chunk: int):
        self.config = config
        self.process = process
        self.transform = transform
        self.action_chunk = action_chunk

    def preprocess(self, obs: dict) -> dict:
        batch = {}
        for key, val in obs.items():
            if key in self.transform:
                val_np = np.array(val)
                if val_np.ndim == 4:  # T, C, H, W
                    frames = [self.transform[key](val_np[t]) for t in range(val_np.shape[0])]
                    batch[key] = torch.stack(frames, dim=0).unsqueeze(0)  # 1, T, C, H, W
                else:  # C, H, W
                    batch[key] = self.transform[key](val_np).unsqueeze(0).unsqueeze(0)  # 1, 1, C, H, W
            elif key in self.process:
                arr = np.array(val, dtype=np.float32).reshape(1, -1) if np.array(val).ndim == 1 else np.array(val, dtype=np.float32)
                # If sequence of vectors: (T, D)
                if arr.ndim == 2 and arr.shape[0] > 1:
                    # transform each timestep? StandardScaler expects (N, D)
                    transformed = self.process[key].transform(arr)
                    t = torch.tensor(transformed, dtype=torch.float32)
                    if key == "action":
                        # Action encoder expects chunked inputs - will be tiled per step later? Here we keep T dimension
                        # For context actions, we keep as is but will be tiled in policy if needed
                        # To match diagram: tiled = t.repeat(1, chunk) for single step, but for sequence we repeat last dim
                        # We'll handle chunk tiling outside, keep simple
                        batch[key] = t.unsqueeze(0)  # 1, T, D  (chunk handled in planner)
                    else:
                        batch[key] = t.unsqueeze(0)  # 1, T, D
                else:
                    arr2 = np.array(val, dtype=np.float32).reshape(1, -1) if arr.ndim == 1 else arr.reshape(1, -1) if arr.ndim == 1 else arr
                    # handle generic
                    if arr.ndim == 1:
                        arr2 = arr.reshape(1, -1)
                    else:
                        arr2 = arr
                        if arr2.ndim == 1:
                            arr2 = arr2.reshape(1, -1)
                        if arr2.ndim > 2:
                            # flatten leading?
                            arr2 = arr2.reshape(-1, arr2.shape[-1])
                    # For single vector case
                    if val.__class__.__name__ != 'ndarray' or np.array(val).ndim <= 2:
                        # single or sequence already handled
                        pass
                    # fallback: try transform
                    try:
                        transformed = self.process[key].transform(np.array(val, dtype=np.float32).reshape(1, -1) if np.array(val).ndim==1 else np.array(val, dtype=np.float32).reshape(-1, np.array(val).shape[-1]) if np.array(val).ndim==2 else np.array(val, dtype=np.float32).reshape(1, -1))
                        # Actually we already did for sequence case above
                    except:
                        pass
                    # Re-do cleanly for both cases:
                    val_np = np.array(val)
                    if val_np.ndim == 1:
                        arr = val_np.reshape(1, -1)
                        transformed = self.process[key].transform(arr)
                        t = torch.tensor(transformed, dtype=torch.float32)
                        if key == "action":
                            tiled = t.repeat(1, self.action_chunk)
                            batch[key] = tiled.unsqueeze(0).unsqueeze(0) if tiled.ndim==2 and tiled.shape[0]==1 else tiled.unsqueeze(0)
                            # ensure shape 1,1,D*chunk
                            if batch[key].ndim == 2:
                                batch[key] = batch[key].unsqueeze(0)
                            if batch[key].ndim == 3 and batch[key].shape[1]!=1:
                                # sequence case
                                pass
                        else:
                            batch[key] = t.unsqueeze(1)
                    elif val_np.ndim == 2:
                        # T, D
                        transformed = self.process[key].transform(val_np)
                        t = torch.tensor(transformed, dtype=torch.float32)
                        if key == "action":
                            # tile each timestep
                            tiled = t.repeat(1, self.action_chunk) if t.shape[0]==1 else np.repeat(t, self.action_chunk, axis=-1) if False else torch.tensor(np.concatenate([np.tile(t[i:i+1].numpy() if isinstance(t, torch.Tensor) else t[i:i+1], self.action_chunk) for i in range(t.shape[0])], axis=0), dtype=torch.float32) if False else t # fallback
                            # Simpler: repeat last dim
                            t_tiled = torch.cat([t]*self.action_chunk, dim=-1) if t.shape[0]>1 else t.repeat(1, self.action_chunk)
                            batch[key] = t_tiled.unsqueeze(0)  # 1, T, D*chunk
                        else:
                            batch[key] = t.unsqueeze(0)
                    else:
                        # unexpected
                        continue
            else:
                if isinstance(val, (np.ndarray, torch.Tensor)):
                    t = torch.as_tensor(val, dtype=torch.float32)
                    while t.ndim < 2:
                        t = t.unsqueeze(0)
                    batch[key] = t
                else:
                    continue
        return batch


# ----------------------------
# Manual Gradient MPC Policy
# Exactly same optimization as diagram.py
# No torch.optim, explicit autograd.grad, energy = sum
# ----------------------------
class DiagramGradPolicy:
    """
    MPC policy that uses the exact manual gradient descent from diagram.py
    for planning. Compatible with swm.World.evaluate.
    """
    def __init__(self, model, process, transform_dict, plan_cfg, grad_cfg, device, keys_to_cache):
        self.model = model
        self.process = process
        self.transform_dict = transform_dict
        self.plan_cfg = plan_cfg
        self.grad_cfg = grad_cfg
        self.device = device
        self.keys_to_cache = keys_to_cache

        self.raw_action_dim = process["action"].mean_.shape[0]
        # infer chunk
        if hasattr(model, "action_encoder") and hasattr(model.action_encoder, "in_features"):
            self.action_chunk = model.action_encoder.in_features // self.raw_action_dim
        elif hasattr(plan_cfg, "action_chunk"):
            self.action_chunk = plan_cfg.action_chunk
        else:
            self.action_chunk = grad_cfg.get("action_chunk", 5)
        self.chunk_dim = self.action_chunk * self.raw_action_dim

        # bounds in normalized space
        self.norm_lo = None
        self.norm_hi = None
        if grad_cfg.get("action_bounds") and "action" in process:
            raw_lo, raw_hi = grad_cfg.action_bounds
            proc = process["action"]
            # proc is StandardScaler, transform needs (1, D)
            self.norm_lo = proc.transform(np.array([[raw_lo]*self.raw_action_dim]))[0, 0]
            self.norm_hi = proc.transform(np.array([[raw_hi]*self.raw_action_dim]))[0, 0]

        # history length
        if hasattr(plan_cfg, "history_size"):
            self.ctx_len = plan_cfg.history_size
        elif hasattr(model, "history_size"):
            self.ctx_len = model.history_size
        else:
            self.ctx_len = 1

        self.horizon = plan_cfg.horizon
        # action_block or receding_horizon
        self.action_block = getattr(plan_cfg, "action_block", getattr(plan_cfg, "receding_horizon", 1))

        self.preprocessor = SingleInstancePreprocessor(
            config=None, process=process, transform=transform_dict, action_chunk=self.action_chunk
        )

        self.obs_buffer = []  # list of dicts raw
        self.act_buffer = []  # list of raw actions
        self.planned_actions = []  # list of decoded raw actions remaining
        self.steps_since_plan = 0
        self.current_goal_raw = None
        self.goal_emb = None

        # gradient hyperparams
        self.lr = grad_cfg.get("lr", 0.1)
        self.n_iter = grad_cfg.get("n_iter", 50)
        self.grad_clip = grad_cfg.get("grad_clip", None)
        self.action_noise = grad_cfg.get("action_noise", 0.0)

    def reset(self, goal=None):
        self.obs_buffer = []
        self.act_buffer = []
        self.planned_actions = []
        self.steps_since_plan = 0
        self.goal_emb = None
        if goal is not None:
            self.current_goal_raw = goal

    def _build_context_data(self):
        """Build context_data dict from buffers with padding like diagram.py"""
        if len(self.obs_buffer) == 0:
            raise ValueError("obs_buffer empty at planning time")
        actual_len = len(self.obs_buffer)
        # pad if needed
        if actual_len < self.ctx_len:
            pad_len = self.ctx_len - actual_len
            # repeat first obs
            padded_obs = [self.obs_buffer[0]]*pad_len + self.obs_buffer
            padded_acts = [self.act_buffer[0] if len(self.act_buffer)>0 else np.zeros(self.raw_action_dim)]*pad_len + self.act_buffer if len(self.act_buffer)>0 else [np.zeros(self.raw_action_dim)]*self.ctx_len
            # ensure act buffer length matches
            if len(padded_acts) < self.ctx_len:
                padded_acts = [np.zeros(self.raw_action_dim)]*(self.ctx_len - len(padded_acts)) + padded_acts
        else:
            padded_obs = self.obs_buffer[-self.ctx_len:]
            padded_acts = self.act_buffer[-self.ctx_len:] if len(self.act_buffer) >= self.ctx_len else [np.zeros(self.raw_action_dim)]*(self.ctx_len - len(self.act_buffer)) + self.act_buffer

        context_data = {}
        # collect keys
        # pixels
        for key in self.keys_to_cache:
            if key == "action":
                # stack actions
                try:
                    arr = np.stack(padded_acts, axis=0)  # T, D
                except:
                    arr = np.array(padded_acts)
                context_data[key] = arr
            else:
                # from obs
                vals = []
                for ob in padded_obs:
                    if key in ob:
                        vals.append(ob[key])
                if len(vals) == 0:
                    continue
                # if images: stack
                try:
                    stacked = np.stack(vals, axis=0)
                except:
                    # fallback for scalar
                    stacked = np.array(vals)
                context_data[key] = stacked
        return context_data

    def _preprocess_and_encode(self, context_data, goal_data):
        proc_context = self.preprocessor.preprocess(context_data)
        proc_goal = self.preprocessor.preprocess(goal_data)
        proc_context = {k: v.to(self.device) for k, v in proc_context.items()}
        proc_goal = {k: v.to(self.device) for k, v in proc_goal.items()}

        with torch.no_grad():
            ctx_out = self.model.encode(proc_context)
            ctx_emb = ctx_out["emb"]  # 1, T_ctx, hidden
            ctx_act = ctx_out["act_emb"]  # 1, T_ctx, embed
            goal_out = self.model.encode(proc_goal)
            goal_emb = goal_out["emb"][:, -1]  # 1, hidden

        return proc_context, proc_goal, ctx_emb, ctx_act, goal_emb

    def _optimize(self, ctx_emb, ctx_act, goal_emb):
        """
        Core optimization loop - EXACTLY as diagram.py
        energy = sum (pred - target)^2  (sum not mean)
        grad via torch.autograd.grad, manual SGD
        """
        device = self.device
        horizon = self.horizon
        chunk_dim = self.chunk_dim

        act_seq = torch.randn(1, horizon, chunk_dim, device=device, requires_grad=True)

        if self.norm_lo is not None and self.norm_hi is not None:
            with torch.no_grad():
                act_seq.clamp_(self.norm_lo, self.norm_hi)

        # optimization
        for i in range(self.n_iter):
            act_emb_seq = self.model.action_encoder(act_seq)  # 1, H, embed

            current_ctx_emb = ctx_emb
            current_ctx_act = ctx_act
            final_pred_emb = None

            for t in range(horizon):
                step_act_emb = act_emb_seq[:, t:t+1]
                full_act_ctx = torch.cat([current_ctx_act[:, 1:], step_act_emb], dim=1)
                pred_out = self.model.predict(current_ctx_emb, full_act_ctx)
                pred_emb = pred_out[:, -1] if pred_out.dim() == 3 else pred_out
                current_ctx_emb = torch.cat([current_ctx_emb[:, 1:], pred_emb.unsqueeze(1)], dim=1)
                current_ctx_act = full_act_ctx
                final_pred_emb = pred_emb

            # energy sum reduction - as in diagram.py
            energy = (final_pred_emb - goal_emb.detach()).pow(2).sum(dim=-1).sum()

            grad_energy = torch.autograd.grad(energy, act_seq, create_graph=False)[0]

            with torch.no_grad():
                if self.grad_clip:
                    grad_norm = grad_energy.norm()
                    if grad_norm > self.grad_clip:
                        grad_energy = grad_energy * (self.grad_clip / (grad_norm + 1e-6))
                act_seq -= self.lr * grad_energy
                if self.action_noise > 0.0:
                    act_seq += self.action_noise * torch.randn_like(act_seq)
                if self.norm_lo is not None:
                    act_seq.clamp_(self.norm_lo, self.norm_hi)

        return act_seq  # 1, H, chunk_dim

    def _decode_actions(self, act_seq_norm):
        # act_seq_norm: 1, H, chunk_dim tensor
        # take first raw_action_dim per step
        raw = act_seq_norm[0].detach().cpu().numpy()  # H, chunk_dim
        # extract first D
        first_chunk = raw[:, :self.raw_action_dim]  # H, D
        # inverse transform via StandardScaler
        inv = self.process["action"].inverse_transform(first_chunk)
        return inv  # H, D

    def _ensure_buffers(self, obs):
        # obs is dict from env, may contain pixels as np array etc
        # Normalize obs keys: expect 'pixels' maybe as (C,H,W) or (H,W,C)
        # Store raw obs
        # For simplicity, store only keys that are in keys_to_cache and not action
        filtered = {}
        for k in self.keys_to_cache:
            if k == "action":
                continue
            if k in obs:
                filtered[k] = obs[k]
            elif k == "pixels" and "observation" in obs:
                filtered[k] = obs["observation"]
        # also support obs being np array (image only)
        if len(filtered) == 0 and isinstance(obs, np.ndarray):
            filtered["pixels"] = obs
        # if obs is dict with 'pixels' key already
        if "pixels" not in filtered and "pixels" in obs:
            filtered["pixels"] = obs["pixels"]
        # fallback: if obs contains 'agent_pos'
        if "agent_pos" in obs:
            filtered["agent_pos"] = obs["agent_pos"]

        self.obs_buffer.append(filtered)
        # keep max size ctx_len (but we pad anyway)
        if len(self.obs_buffer) > self.ctx_len*2:  # keep some history
            self.obs_buffer = self.obs_buffer[-self.ctx_len:]

    def get_action(self, obs):
        """
        Called by swm.World. obs may be:
        - dict with 'pixels', 'goal', 'agent_pos' etc
        - or tuple (obs, info)
        We handle dict.
        """
        # obs can be batched? Assume single
        # Extract goal if present
        goal_raw = None
        if isinstance(obs, dict):
            if "goal" in obs:
                goal_raw = obs["goal"]
                # keep current obs without goal for buffer
                obs_no_goal = {k: v for k, v in obs.items() if k != "goal"}
            else:
                obs_no_goal = obs
        else:
            obs_no_goal = {"pixels": obs}
            goal_raw = None

        # If we have new goal, cache it
        if goal_raw is not None:
            self.current_goal_raw = goal_raw
            self.goal_emb = None  # will recompute

        # If we have planned actions remaining and not time to replan, return next
        if len(self.planned_actions) > 0 and self.steps_since_plan < self.action_block:
            action = self.planned_actions.pop(0)
            self.act_buffer.append(action)
            self.steps_since_plan += 1
            # update obs buffer with latest obs (for next planning context, we already have current obs? Actually obs for next step will come next call)
            # We add current obs now to have it for next plan
            self._ensure_buffers(obs_no_goal)
            return action

        # Need to replan
        # Ensure current obs is in buffer
        self._ensure_buffers(obs_no_goal)

        context_data = self._build_context_data()
        # goal data
        if self.current_goal_raw is None:
            # fallback: use last obs as goal? Should not happen
            goal_data = {"pixels": self.obs_buffer[-1].get("pixels")}
        else:
            goal_data = {"pixels": self.current_goal_raw}
            # if goal_raw is dict with pixels
            if isinstance(self.current_goal_raw, dict) and "pixels" in self.current_goal_raw:
                goal_data = self.current_goal_raw
            elif isinstance(self.current_goal_raw, np.ndarray):
                goal_data = {"pixels": self.current_goal_raw}

        # encode
        _, _, ctx_emb, ctx_act, goal_emb = self._preprocess_and_encode(context_data, goal_data)

        # optimize
        act_seq_norm = self._optimize(ctx_emb, ctx_act, goal_emb)

        decoded_horizon = self._decode_actions(act_seq_norm)  # H, D

        # store planned actions
        self.planned_actions = [decoded_horizon[i] for i in range(self.horizon)]
        self.steps_since_plan = 0

        # pop first
        action = self.planned_actions.pop(0)
        self.act_buffer.append(action)
        self.steps_since_plan = 1

        return action

    # aliases for compatibility
    def act(self, obs):
        return self.get_action(obs)

    def __call__(self, obs):
        return self.get_action(obs)


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, \
        "Planning horizon * action_block must be <= eval_budget"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # World
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    # force single env for deterministic evaluation
    if "num_envs" not in cfg.world:
        cfg.world.num_envs = 1
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # transforms
    transform_dict = {
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

    # Load model exactly like diagram.py
    policy_name = cfg.get("policy", "random")
    if policy_name == "random":
        print("Using random policy - no model")
        policy = swm.policy.RandomPolicy()
    else:
        ckpt_path = cfg.policy + "_object.ckpt"
        print(f"Loading model object from {ckpt_path}...")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        plan_cfg = swm.PlanConfig(**cfg.plan_config)
        grad_cfg = OmegaConf.to_container(cfg.get("gradient_solver", cfg.get("solver", {})), resolve=True) if cfg.get("gradient_solver") or cfg.get("solver") else {}
        # ensure n_iter etc from diagram defaults
        # if user passed gradient_solver in yaml, use it, else use diagram defaults + cfg
        grad_cfg.setdefault("n_iter", 50)
        grad_cfg.setdefault("lr", 0.1)
        grad_cfg.setdefault("grad_clip", None)
        grad_cfg.setdefault("action_noise", 0.0)
        grad_cfg.setdefault("action_bounds", None)
        grad_cfg.setdefault("action_chunk", 5)

        print(f"Plan config: horizon={plan_cfg.horizon}, action_block={plan_cfg.action_block}, history={getattr(plan_cfg,'history_size', 'model default')}")
        print(f"Grad config: {grad_cfg}")

        policy = DiagramGradPolicy(
            model=model,
            process=process,
            transform_dict=transform_dict,
            plan_cfg=plan_cfg,
            grad_cfg=grad_cfg,
            device=device,
            keys_to_cache=cfg.dataset.keys_to_cache
        )

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample episodes like original eval.py
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print(f"Sampled indices: {random_episode_indices}")

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

    print("\n==== METRICS ====")
    print(metrics)
    success_rate = metrics.get("success_rate", metrics.get("success", 0))
    print(f"\nSuccess Rate: {success_rate}")

    results_file = results_path / cfg.output.filename
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
