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

# ---------------------------------------------------------------------------
# PATCH for GradientSolver device bug (from your working script)
# ---------------------------------------------------------------------------
try:
    from stable_worldmodel.solver.gd import GradientSolver
    def _patched_init_action(self, n_envs, actions=None):
        if actions is None:
            actions = torch.zeros((n_envs, 0, self.action_dim), dtype=self.dtype)
        remaining = self.horizon - actions.shape[1]
        if remaining > 0:
            new_actions = torch.zeros(n_envs, remaining, self.action_dim, dtype=self.dtype)
            actions = torch.cat([actions, new_actions], dim=1)
        actions = actions.to(self.device)  # always move
        actions = actions.unsqueeze(1).repeat_interleave(self.num_samples, dim=1)
        actions[:, 1:] += (
            torch.randn(
                actions[:, 1:].shape,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )
            * self.var_scale
        )
        if hasattr(self, "init") and self.init.shape == actions.shape:
            self.init.copy_(actions)
        else:
            if "init" in self._parameters:
                del self._parameters["init"]
            self.register_parameter("init", torch.nn.Parameter(actions))
    GradientSolver.init_action = _patched_init_action
    print("Patched GradientSolver.init_action for device bug")
except Exception as e:
    print(f"Patch skipped: {e}")

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
    def __init__(self, process: dict, transform: dict, action_chunk: int):
        self.process = process
        self.transform = transform
        self.action_chunk = action_chunk
    def preprocess(self, obs: dict) -> dict:
        batch = {}
        for key, val in obs.items():
            val_np = np.array(val)
            if key in self.transform:
                if val_np.ndim == 4:
                    frames = [self.transform[key](val_np[t]) for t in range(val_np.shape[0])]
                    batch[key] = torch.stack(frames, dim=0).unsqueeze(0)
                elif val_np.ndim == 3:
                    batch[key] = self.transform[key](val_np).unsqueeze(0).unsqueeze(0)
                else:
                    batch[key] = self.transform[key](val_np).unsqueeze(0).unsqueeze(0)
            elif key in self.process:
                if val_np.ndim == 1:
                    arr = val_np.reshape(1, -1)
                    trans = self.process[key].transform(arr)
                    t = torch.tensor(trans, dtype=torch.float32)
                    if key == "action":
                        tiled = t.repeat(1, self.action_chunk)
                        batch[key] = tiled.unsqueeze(1)
                    else:
                        batch[key] = t.unsqueeze(1)
                elif val_np.ndim == 2:
                    trans = self.process[key].transform(val_np)
                    t = torch.tensor(trans, dtype=torch.float32)
                    if key == "action":
                        tiled = torch.cat([t]*self.action_chunk, dim=-1) if t.shape[0] > 1 else t.repeat(1, self.action_chunk)
                        batch[key] = tiled.unsqueeze(0)
                    else:
                        batch[key] = t.unsqueeze(0)
        return batch


# ---------------------------------------------------------------------------
# EXACT DIAGRAM SOLVER - compatible with WorldModelPolicy
# This solver implements the exact manual GD from diagram.py
# energy = sum (not mean), torch.autograd.grad, no torch.optim
# ---------------------------------------------------------------------------
class DiagramSolver(torch.nn.Module):
    """
    Custom solver that WorldModelPolicy can call.
    It mimics the API of GradientSolver but does exact diagram optimization.
    """
    def __init__(self, model, device="cuda", horizon=5, action_dim=2, 
                 lr=0.1, n_iter=50, grad_clip=None, action_noise=0.0, 
                 action_bounds=None, action_chunk=5, process=None, **kwargs):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.horizon = horizon
        self.action_dim = action_dim
        self.lr = lr
        self.n_iter = n_iter
        self.grad_clip = grad_clip
        self.action_noise = action_noise
        self.action_bounds = action_bounds
        self.action_chunk = action_chunk
        self.process = process
        self.chunk_dim = action_chunk * action_dim
        self.dtype = torch.float32

        # for compatibility with WorldModelPolicy that expects these attrs
        self.num_samples = 1
        self.var_scale = 0.0
        self.torch_gen = torch.Generator(device=self.device)
        self.torch_gen.manual_seed(42)

        # bounds in norm space
        self.norm_lo = None
        self.norm_hi = None
        if action_bounds and process and "action" in process:
            raw_lo, raw_hi = action_bounds
            proc = process["action"]
            self.norm_lo = proc.transform(np.array([[raw_lo]*action_dim]))[0,0]
            self.norm_hi = proc.transform(np.array([[raw_hi]*action_dim]))[0,0]

        # dummy parameter for compatibility with patched init_action
        self.register_parameter("init", torch.nn.Parameter(torch.zeros(1,1,horizon,action_dim)))

    def to(self, *args, **kwargs):
        # keep model on device, but allow nn.Module.to call
        return super().to(*args, **kwargs)

    # WorldModelPolicy may call solver.init_action - provide compatible version
    def init_action(self, n_envs, actions=None):
        if actions is None:
            actions = torch.zeros((n_envs, 0, self.action_dim), dtype=self.dtype, device=self.device)
        remaining = self.horizon - actions.shape[1]
        if remaining > 0:
            new_actions = torch.zeros(n_envs, remaining, self.action_dim, dtype=self.dtype, device=self.device)
            actions = torch.cat([actions, new_actions], dim=1)
        actions = actions.to(self.device)
        return actions

    def solve(self, obs_emb, act_emb, goal_emb):
        """
        obs_emb: (B, T_ctx, D), act_emb: (B, T_ctx, D_act), goal_emb: (B, D)
        Returns: best action sequence (B, H, D_raw) ??? WorldModelPolicy expects (B, H, D)
        We implement exact diagram loop batched for B=1 case, but handle B>1 by looping.
        """
        B = obs_emb.shape[0]
        device = self.device
        all_best = []

        for b in range(B):
            ctx_emb = obs_emb[b:b+1]  # 1,T,D
            ctx_act = act_emb[b:b+1]
            g_emb = goal_emb[b:b+1]

            # init random action seq - EXACTLY as diagram.py
            act_seq = torch.randn(1, self.horizon, self.chunk_dim, device=device, requires_grad=True)
            if self.norm_lo is not None:
                with torch.no_grad():
                    act_seq.clamp_(self.norm_lo, self.norm_hi)

            for _ in range(self.n_iter):
                act_emb_seq = self.model.action_encoder(act_seq)  # 1,H,embed
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

                # SUM reduction - critical
                energy = (final_pred - g_emb.detach()).pow(2).sum(dim=-1).sum()
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

            # decode: take first raw_action_dim per step
            raw = act_seq[0].detach().cpu().numpy()  # H, chunk
            first = raw[:, :self.action_dim]
            # inverse transform
            if self.process is not None and "action" in self.process:
                inv = self.process["action"].inverse_transform(first)
            else:
                inv = first
            all_best.append(inv)

        # (B, H, D)
        return torch.tensor(np.stack(all_best, axis=0), device=device, dtype=torch.float32)

    # Some WorldModelPolicy versions call .plan or .forward
    def plan(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.solve(*args, **kwargs)


# ---------------------------------------------------------------------------
# Policy wrapper that HAS set_env - required by World.set_policy
# Inherits from WorldModelPolicy to get all compatibility
# ---------------------------------------------------------------------------
class DiagramPolicy(swm.policy.WorldModelPolicy):
    def __init__(self, model, process, transform, plan_cfg, grad_cfg, device):
        # Build exact diagram solver
        raw_dim = process["action"].mean_.shape[0]
        solver = DiagramSolver(
            model=model,
            device=device,
            horizon=plan_cfg.horizon,
            action_dim=raw_dim,
            lr=grad_cfg.get("lr", 0.1),
            n_iter=grad_cfg.get("n_iter", 50),
            grad_clip=grad_cfg.get("grad_clip", None),
            action_noise=grad_cfg.get("action_noise", 0.0),
            action_bounds=grad_cfg.get("action_bounds", None),
            action_chunk=grad_cfg.get("action_chunk", 5),
            process=process,
        )
        super().__init__(solver=solver, config=plan_cfg, process=process, transform=transform)
        self.diagram_model = model
        self.device = device


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    if "num_envs" not in cfg.world:
        cfg.world.num_envs = 1

    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
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
        ckpt_path = cfg.policy
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path += "_object.ckpt" if "_object" not in ckpt_path else ".ckpt"
            # handle case where policy is full path without suffix handling
            if not os.path.exists(ckpt_path):
                # try as given + .ckpt
                alt = cfg.policy + "_object.ckpt"
                if os.path.exists(alt):
                    ckpt_path = alt
                else:
                    ckpt_path = cfg.policy + ".ckpt"
        # if user passed full path to _object.ckpt, use as is
        if os.path.exists(cfg.policy):
            ckpt_path = cfg.policy
        if os.path.exists(cfg.policy + ".ckpt"):
            ckpt_path = cfg.policy + ".ckpt"
        if os.path.exists(cfg.policy + "_object.ckpt"):
            ckpt_path = cfg.policy + "_object.ckpt"

        print(f"Loading {ckpt_path}")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model.to("cuda")
        model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        plan_cfg = swm.PlanConfig(**cfg.plan_config)
        # get grad cfg from config - support both solver and gradient_solver keys
        grad_cfg = {}
        if cfg.get("gradient_solver"):
            grad_cfg = OmegaConf.to_container(cfg.gradient_solver, resolve=True)
        elif cfg.get("solver"):
            # solver may contain optimizer stuff, we extract relevant
            s = OmegaConf.to_container(cfg.solver, resolve=True)
            grad_cfg = {k: s.get(k, v) for k, v in {
                "n_iter": 50, "lr": 0.1, "grad_clip": None, "action_noise": 0.0,
                "action_bounds": None, "action_chunk": 5
            }.items()}
            # override with values if present
            for k in ["n_iter", "lr", "grad_clip", "action_noise", "action_bounds", "action_chunk"]:
                if k in s:
                    grad_cfg[k] = s[k]
        grad_cfg.setdefault("n_iter", 50)
        grad_cfg.setdefault("lr", 0.1)
        grad_cfg.setdefault("grad_clip", None)
        grad_cfg.setdefault("action_noise", 0.0)
        grad_cfg.setdefault("action_bounds", None)
        grad_cfg.setdefault("action_chunk", 5)

        print(f"Plan: horizon={plan_cfg.horizon} block={plan_cfg.action_block}")
        print(f"Grad: {grad_cfg}")

        policy = DiagramPolicy(
            model=model,
            process=process,
            transform=transform,
            plan_cfg=plan_cfg,
            grad_cfg=grad_cfg,
            device=device,
        )

    # sample valid starts - same as original eval.py
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
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_starts.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_eps.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    print(metrics)

    with (results_path / cfg.output.filename).open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\nmetrics: {metrics}\n")
        f.write(f"time: {time.time()-start}\n")


if __name__ == "__main__":
    run()
