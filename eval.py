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


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


class SingleInstancePreprocessor:
    """Clean version - same logic as diagram.py, supports T sequences"""
    def __init__(self, process: dict, transform: dict, action_chunk: int):
        self.process = process
        self.transform = transform
        self.action_chunk = action_chunk

    def preprocess(self, obs: dict) -> dict:
        batch = {}
        for key, val in obs.items():
            val_np = np.array(val)
            if key in self.transform:
                if val_np.ndim == 4:  # T,C,H,W
                    frames = [self.transform[key](val_np[t]) for t in range(val_np.shape[0])]
                    batch[key] = torch.stack(frames, dim=0).unsqueeze(0)
                elif val_np.ndim == 3:  # C,H,W
                    batch[key] = self.transform[key](val_np).unsqueeze(0).unsqueeze(0)
                else:
                    # H,W,C -> ToImage will handle
                    batch[key] = self.transform[key](val_np).unsqueeze(0).unsqueeze(0)
            elif key in self.process:
                # vector data
                if val_np.ndim == 1:  # D
                    arr = val_np.reshape(1, -1)
                    trans = self.process[key].transform(arr)
                    t = torch.tensor(trans, dtype=torch.float32)
                    if key == "action":
                        tiled = t.repeat(1, self.action_chunk)
                        batch[key] = tiled.unsqueeze(1)  # 1,1,D*chunk
                    else:
                        batch[key] = t.unsqueeze(1)  # 1,1,D
                elif val_np.ndim == 2:  # T,D
                    trans = self.process[key].transform(val_np)
                    t = torch.tensor(trans, dtype=torch.float32)  # T,D
                    if key == "action":
                        # tile each timestep's action
                        tiled = torch.cat([t]*self.action_chunk, dim=-1) if t.shape[0] > 1 else t.repeat(1, self.action_chunk)
                        batch[key] = tiled.unsqueeze(0)  # 1,T,D*chunk
                    else:
                        batch[key] = t.unsqueeze(0)  # 1,T,D
                else:
                    continue
        return batch


class DiagramGradPolicy:
    """
    Exact manual GD from diagram.py:
    - energy = sum (pred-goal)^2  (SUM not mean)
    - torch.autograd.grad
    - manual SGD, no torch.optim
    - grad clip, noise, bounds
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
        if hasattr(model, "action_encoder") and hasattr(model.action_encoder, "in_features"):
            self.action_chunk = model.action_encoder.in_features // self.raw_action_dim
        else:
            self.action_chunk = grad_cfg.get("action_chunk", 5)
        self.chunk_dim = self.action_chunk * self.raw_action_dim

        # bounds
        self.norm_lo = None
        self.norm_hi = None
        if grad_cfg.get("action_bounds") and "action" in process:
            raw_lo, raw_hi = grad_cfg.action_bounds
            proc = process["action"]
            self.norm_lo = proc.transform(np.array([[raw_lo]*self.raw_action_dim]))[0,0]
            self.norm_hi = proc.transform(np.array([[raw_hi]*self.raw_action_dim]))[0,0]

        self.ctx_len = getattr(plan_cfg, "history_size", getattr(model, "history_size", 1))
        self.horizon = plan_cfg.horizon
        self.action_block = getattr(plan_cfg, "action_block", getattr(plan_cfg, "receding_horizon", 1))

        self.preprocessor = SingleInstancePreprocessor(process, transform_dict, self.action_chunk)

        self.obs_buffer = []
        self.act_buffer = []
        self.planned_actions = []
        self.steps_since_plan = 0
        self.current_goal_raw = None

        self.lr = grad_cfg.get("lr", 0.1)
        self.n_iter = grad_cfg.get("n_iter", 50)
        self.grad_clip = grad_cfg.get("grad_clip", None)
        self.action_noise = grad_cfg.get("action_noise", 0.0)

    def reset(self, goal=None):
        self.obs_buffer = []
        self.act_buffer = []
        self.planned_actions = []
        self.steps_since_plan = 0
        if goal is not None:
            self.current_goal_raw = goal

    def _build_context(self):
        if len(self.obs_buffer) == 0:
            raise ValueError("empty buffer")
        # pad to ctx_len
        if len(self.obs_buffer) < self.ctx_len:
            pad = self.ctx_len - len(self.obs_buffer)
            obs_padded = [self.obs_buffer[0]]*pad + self.obs_buffer
            act_padded = ([self.act_buffer[0]] if self.act_buffer else [np.zeros(self.raw_action_dim)])*pad + self.act_buffer if self.act_buffer else [np.zeros(self.raw_action_dim)]*self.ctx_len
            if len(act_padded) < self.ctx_len:
                act_padded = [np.zeros(self.raw_action_dim)]*(self.ctx_len - len(act_padded)) + act_padded
        else:
            obs_padded = self.obs_buffer[-self.ctx_len:]
            if len(self.act_buffer) >= self.ctx_len:
                act_padded = self.act_buffer[-self.ctx_len:]
            else:
                act_padded = [np.zeros(self.raw_action_dim)]*(self.ctx_len - len(self.act_buffer)) + self.act_buffer

        ctx = {}
        for k in self.keys_to_cache:
            if k == "action":
                ctx[k] = np.stack(act_padded, axis=0)
            else:
                vals = [o[k] for o in obs_padded if k in o]
                if not vals:
                    continue
                try:
                    ctx[k] = np.stack(vals, axis=0)
                except:
                    ctx[k] = np.array(vals)
        return ctx

    def _optimize(self, ctx_emb, ctx_act, goal_emb):
        device = self.device
        act_seq = torch.randn(1, self.horizon, self.chunk_dim, device=device, requires_grad=True)
        if self.norm_lo is not None:
            with torch.no_grad():
                act_seq.clamp_(self.norm_lo, self.norm_hi)

        for _ in range(self.n_iter):
            act_emb_seq = self.model.action_encoder(act_seq)
            cur_emb = ctx_emb
            cur_act = ctx_act
            final_pred = None
            for t in range(self.horizon):
                step_act = act_emb_seq[:, t:t+1]
                full_act = torch.cat([cur_act[:, 1:], step_act], dim=1)
                pred = self.model.predict(cur_emb, full_act)
                pred = pred[:, -1] if pred.dim() == 3 else pred
                cur_emb = torch.cat([cur_emb[:, 1:], pred.unsqueeze(1)], dim=1)
                cur_act = full_act
                final_pred = pred

            # SUM reduction as in diagram.py
            energy = (final_pred - goal_emb.detach()).pow(2).sum(dim=-1).sum()
            grad = torch.autograd.grad(energy, act_seq, create_graph=False)[0]

            with torch.no_grad():
                if self.grad_clip:
                    gn = grad.norm()
                    if gn > self.grad_clip:
                        grad = grad * (self.grad_clip / (gn + 1e-6))
                act_seq -= self.lr * grad
                if self.action_noise > 0:
                    act_seq += self.action_noise * torch.randn_like(act_seq)
                if self.norm_lo is not None:
                    act_seq.clamp_(self.norm_lo, self.norm_hi)
        return act_seq

    def _decode(self, act_seq):
        raw = act_seq[0].detach().cpu().numpy()  # H, chunk
        first = raw[:, :self.raw_action_dim]
        return self.process["action"].inverse_transform(first)

    def _ensure_obs(self, obs_no_goal):
        filt = {}
        for k in self.keys_to_cache:
            if k == "action":
                continue
            if k in obs_no_goal:
                filt[k] = obs_no_goal[k]
        if not filt and isinstance(obs_no_goal, dict) and "pixels" in obs_no_goal:
            filt["pixels"] = obs_no_goal["pixels"]
        elif isinstance(obs_no_goal, np.ndarray):
            filt["pixels"] = obs_no_goal
        # common fallback for pusht: obs may contain 'observation' or be dict with state
        if "pixels" not in filt and "observation" in obs_no_goal:
            filt["pixels"] = obs_no_goal["observation"]
        self.obs_buffer.append(filt)
        if len(self.obs_buffer) > 100:
            self.obs_buffer = self.obs_buffer[-self.ctx_len:]

    def get_action(self, obs):
        # obs dict may contain goal
        if isinstance(obs, dict):
            goal = obs.get("goal", None)
            obs_no_goal = {k: v for k, v in obs.items() if k != "goal"}
        else:
            goal = None
            obs_no_goal = {"pixels": obs}

        if goal is not None:
            self.current_goal_raw = goal

        # return cached plan if available
        if self.planned_actions and self.steps_since_plan < self.action_block:
            a = self.planned_actions.pop(0)
            self.act_buffer.append(a)
            self.steps_since_plan += 1
            self._ensure_obs(obs_no_goal)
            return a

        # replan
        self._ensure_obs(obs_no_goal)
        ctx_data = self._build_context()

        if self.current_goal_raw is None:
            goal_data = {"pixels": self.obs_buffer[-1]["pixels"]}
        else:
            g = self.current_goal_raw
            if isinstance(g, dict) and "pixels" in g:
                goal_data = g
            elif isinstance(g, np.ndarray):
                goal_data = {"pixels": g}
            else:
                goal_data = {"pixels": g}

        proc_ctx = self.preprocessor.preprocess(ctx_data)
        proc_goal = self.preprocessor.preprocess(goal_data)
        proc_ctx = {k: v.to(self.device) for k, v in proc_ctx.items()}
        proc_goal = {k: v.to(self.device) for k, v in proc_goal.items()}

        with torch.no_grad():
            ctx_out = self.model.encode(proc_ctx)
            ctx_emb = ctx_out["emb"]
            ctx_act = ctx_out["act_emb"]
            goal_out = self.model.encode(proc_goal)
            goal_emb = goal_out["emb"][:, -1]

        act_seq = self._optimize(ctx_emb, ctx_act, goal_emb)
        decoded = self._decode(act_seq)

        self.planned_actions = [decoded[i] for i in range(self.horizon)]
        self.steps_since_plan = 0
        a = self.planned_actions.pop(0)
        self.act_buffer.append(a)
        self.steps_since_plan = 1
        return a

    def act(self, obs): return self.get_action(obs)
    def __call__(self, obs): return self.get_action(obs)


def try_world_evaluate(world, dataset, start_steps, goal_offset, eval_budget, episodes_idx, callables, video_path):
    """Try multiple API names for compatibility"""
    # newer versions: evaluate, evaluate_from_dataset, _evaluate_from_dataset
    for name in ["evaluate", "evaluate_from_dataset", "_evaluate_from_dataset", "evaluate_dataset"]:
        if hasattr(world, name):
            fn = getattr(world, name)
            try:
                print(f"Trying World.{name} ...")
                # inspect signature: try with dataset kwargs, fallback without
                try:
                    return fn(
                        dataset=dataset,
                        start_steps=start_steps,
                        goal_offset=goal_offset,
                        eval_budget=eval_budget,
                        episodes_idx=episodes_idx,
                        callables=callables,
                        video=video_path,
                    )
                except TypeError as e:
                    print(f"{name} TypeError with dataset args: {e}, trying without dataset args")
                    # some versions just need episodes
                    return fn(episodes=len(episodes_idx))
            except Exception as e:
                print(f"{name} failed: {e}")
                continue
    return None


def manual_evaluate(world, policy, dataset, start_indices, goal_offset, eval_budget, episodes_idx):
    """Fallback manual loop when World.evaluate API is missing - computes success rate"""
    print("\n[Fallback] Running manual evaluation loop...")
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col = dataset.get_col_data(col_name)
    step_col = dataset.get_col_data("step_idx")

    # try to get env
    # World may have .envs list or .env
    def get_env(w):
        for attr in ["env", "envs", "_env", "environment", "vec_env"]:
            if hasattr(w, attr):
                e = getattr(w, attr)
                if isinstance(e, (list, tuple)):
                    return e[0]
                return e
        # try gymnasium directly
        import gymnasium as gym
        return gym.make("swm/PushT-v1", render_mode="rgb_array")

    env = get_env(world)
    print(f"Using env: {env}")

    successes = 0
    total = len(start_indices)

    for i in range(total):
        start_abs_idx = start_indices[i]
        ep_id = episodes_idx[i]

        # find goal abs idx
        ep_mask = ep_col == ep_id
        # start step
        start_step = step_col[start_abs_idx]
        goal_step = start_step + goal_offset
        # find goal idx in this episode
        candidates = np.where(ep_mask & (step_col == goal_step))[0]
        if len(candidates) == 0:
            # clamp to max
            max_step = np.max(step_col[ep_mask])
            goal_step = min(goal_step, max_step)
            candidates = np.where(ep_mask & (step_col == goal_step))[0]
        goal_abs_idx = candidates[0]

        start_row = dataset.get_row_data(start_abs_idx)
        goal_row = dataset.get_row_data(goal_abs_idx)

        goal_pixels = goal_row["pixels"]

        # reset env
        try:
            obs, info = env.reset()
        except:
            obs = env.reset()

        # Try to set state if dataset has 'state'
        if "state" in start_row:
            try:
                if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "set_state"):
                    env.unwrapped.set_state(start_row["state"])
                    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else obs
                elif hasattr(env, "set_state"):
                    env.set_state(start_row["state"])
            except Exception as e:
                pass

        policy.reset(goal=goal_pixels)

        # if obs is dict, extract pixels
        if isinstance(obs, dict):
            cur_pixels = obs.get("pixels", obs.get("observation", None))
        else:
            cur_pixels = obs

        # rollout
        done = False
        for t in range(eval_budget):
            # policy expects dict with pixels and goal
            act = policy.get_action({"pixels": cur_pixels, "goal": goal_pixels})
            # env step
            try:
                nxt, rew, term, trunc, info = env.step(act)
                done = term or trunc
            except ValueError:
                nxt, rew, done, info = env.step(act)

            # success check from info
            if isinstance(info, dict) and (info.get("success") or info.get("is_success") or info.get("success_rate") or rew > 0.9):
                successes += 1
                break

            if isinstance(nxt, dict):
                cur_pixels = nxt.get("pixels", nxt.get("observation", nxt))
            else:
                cur_pixels = nxt

            # for pusht, we can also check distance if state available
            # heuristic: if env has get_state, compare block pos
            if t == eval_budget - 1:
                # final step - check if close to goal using info
                # fallback: assume not success if not flagged
                pass

            if done:
                # check one more time
                if isinstance(info, dict) and info.get("success"):
                    successes += 1
                break

        if (i+1) % 5 == 0:
            print(f"  {i+1}/{total} success so far: {successes}/{i+1} = {successes/(i+1):.2f}")

    return {"success_rate": successes/total, "num_success": successes, "num_eval": total}


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    if "num_envs" not in cfg.world:
        cfg.world.num_envs = 1

    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform_dict = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

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

    policy_name = cfg.get("policy", "random")
    if policy_name == "random":
        policy = swm.policy.RandomPolicy()
    else:
        ckpt_path = cfg.policy + "_object.ckpt"
        print(f"Loading {ckpt_path}")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        plan_cfg = swm.PlanConfig(**cfg.plan_config)
        grad_cfg = OmegaConf.to_container(cfg.get("gradient_solver", cfg.get("solver", {})), resolve=True) if cfg.get("gradient_solver") or cfg.get("solver") else {}
        grad_cfg.setdefault("n_iter", 50)
        grad_cfg.setdefault("lr", 0.1)
        grad_cfg.setdefault("grad_clip", None)
        grad_cfg.setdefault("action_noise", 0.0)
        grad_cfg.setdefault("action_bounds", None)
        grad_cfg.setdefault("action_chunk", 5)

        print(f"Plan: horizon={plan_cfg.horizon} block={plan_cfg.action_block} hist={getattr(plan_cfg,'history_size', 'model')}")
        print(f"Grad: {grad_cfg}")

        policy = DiagramGradPolicy(model, process, transform_dict, plan_cfg, grad_cfg, device, cfg.dataset.keys_to_cache)

    # sample valid starts
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = episode_len - cfg.eval.goal_offset_steps - 1
    max_dict = {ep_id: max_start[i] for i, ep_id in enumerate(ep_indices)}
    max_per_row = np.array([max_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starts")

    rng = np.random.default_rng(cfg.seed)
    sampled = rng.choice(len(valid_indices)-1, size=cfg.eval.num_eval, replace=False)
    sampled = np.sort(valid_indices[sampled])

    eval_eps = dataset.get_row_data(sampled)[col_name]
    eval_starts = dataset.get_row_data(sampled)["step_idx"]

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent if cfg.policy != "random" else Path(__file__).parent
    results_path.mkdir(parents=True, exist_ok=True)

    world.set_policy(policy)

    start = time.time()
    metrics = try_world_evaluate(
        world, dataset, eval_starts.tolist(), cfg.eval.goal_offset_steps,
        cfg.eval.eval_budget, eval_eps.tolist(),
        OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        results_path
    )

    if metrics is None:
        metrics = manual_evaluate(world, policy, dataset, sampled, cfg.eval.goal_offset_steps, cfg.eval.eval_budget, eval_eps.tolist())

    print("\n=== METRICS ===")
    print(metrics)

    with (results_path / cfg.output.filename).open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\nmetrics: {metrics}\n")
        f.write(f"time: {time.time()-start}\n")


if __name__ == "__main__":
    run()
