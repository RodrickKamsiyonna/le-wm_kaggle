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
from stable_worldmodel.solver.gd import GradientSolver

def _patched_init_action(self, n_envs, actions=None):
    if actions is None:
        actions = torch.zeros((n_envs, 0, self.action_dim), dtype=self.dtype)
    remaining = self.horizon - actions.shape[1]
    if remaining > 0:
        new_actions = torch.zeros(n_envs, remaining, self.action_dim, dtype=self.dtype)
        actions = torch.cat([actions, new_actions], dim=1)
    actions = actions.to(self.device)
    actions = actions.unsqueeze(1).repeat_interleave(self.num_samples, dim=1)
    actions[:, 1:] += (torch.randn(actions[:, 1:].shape, generator=self.torch_gen, device=self.device, dtype=self.dtype) * self.var_scale)
    if hasattr(self, "init") and self.init.shape == actions.shape:
        self.init.copy_(actions)
    else:
        if "init" in self._parameters:
            del self._parameters["init"]
        self.register_parameter("init", torch.nn.Parameter(actions))
GradientSolver.init_action = _patched_init_action

class DiagramSolver(GradientSolver):
    def __init__(self, model, device="cuda", horizon=5, action_dim=2, lr=0.1, n_iter=50, grad_clip=None, action_noise=0.0, action_bounds=None, action_chunk=5, process=None, **kwargs):
        try:
            super().__init__(model=model, device=device, action_dim=action_dim, **kwargs)
        except TypeError:
            super().__init__(model=model, device=device, action_dim=action_dim)
        try:
            self.horizon = horizon
        except AttributeError:
            object.__setattr__(self, "_horizon", int(horizon))
        self.action_chunk = action_chunk
        self.chunk_dim = action_chunk * action_dim
        self.lr = lr
        self.n_iter = n_iter
        self.grad_clip = grad_clip
        self.action_noise = action_noise
        self.action_bounds = action_bounds
        self.process = process
        self.norm_lo = None
        self.norm_hi = None
        if action_bounds and process and "action" in process:
            raw_lo, raw_hi = action_bounds
            proc = process["action"]
            self.norm_lo = proc.transform(np.array([[raw_lo]*action_dim]))[0,0]
            self.norm_hi = proc.transform(np.array([[raw_hi]*action_dim]))[0,0]
    def solve(self, obs_emb, act_emb, goal_emb):
        B = obs_emb.shape[0]
        device = self.device
        all_best = []
        for b in range(B):
            ctx_emb = obs_emb[b:b+1]
            ctx_act = act_emb[b:b+1]
            g_emb = goal_emb[b:b+1]
            act_seq = torch.randn(1, self.horizon, self.chunk_dim, device=device, requires_grad=True)
            if self.norm_lo is not None:
                with torch.no_grad(): act_seq.clamp_(self.norm_lo, self.norm_hi)
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
            raw_norm = act_seq[0].detach().cpu().numpy()
            first_norm = raw_norm[:, :self.action_dim]
            all_best.append(first_norm)
        return torch.tensor(np.stack(all_best, axis=0), device=device, dtype=torch.float32)

def img_transform(cfg):
    return transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size=cfg.eval.img_size)])
def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])
def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dataset_path)

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    print("Device: cuda")
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    if "num_envs" not in cfg.world: cfg.world.num_envs = 1
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]: continue
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
        if col != "action": process[f"goal_{col}"] = scaler
    ckpt_path = cfg.policy
    for cand in [cfg.policy, cfg.policy + "_object.ckpt", cfg.policy + ".ckpt"]:
        if os.path.exists(cand):
            ckpt_path = cand
            break
    if not os.path.exists(ckpt_path):
        ckpt_path = cfg.policy + "_object.ckpt"
    print(f"Loading {ckpt_path}")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = model.to("cuda")
    model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    plan_cfg = swm.PlanConfig(**cfg.plan_config)
    grad_cfg = {}
    if cfg.get("gradient_solver"):
        grad_cfg = OmegaConf.to_container(cfg.gradient_solver, resolve=True)
    elif cfg.get("solver"):
        s = OmegaConf.to_container(cfg.solver, resolve=True)
        for k in ["n_iter","lr","grad_clip","action_noise","action_bounds","action_chunk"]:
            if k in s: grad_cfg[k] = s[k]
    grad_cfg.setdefault("n_iter", 50)
    grad_cfg.setdefault("lr", 0.1)
    grad_cfg.setdefault("grad_clip", None)
    grad_cfg.setdefault("action_noise", 0.0)
    grad_cfg.setdefault("action_bounds", None)
    grad_cfg.setdefault("action_chunk", 5)
    print(f"Plan: horizon={plan_cfg.horizon} block={plan_cfg.action_block} Grad: {grad_cfg}")
    solver = DiagramSolver(model=model, device="cuda", horizon=plan_cfg.horizon, action_dim=process["action"].mean_.shape[0], lr=grad_cfg.get("lr", 0.1), n_iter=grad_cfg.get("n_iter", 50), grad_clip=grad_cfg.get("grad_clip"), action_noise=grad_cfg.get("action_noise", 0.0), action_bounds=grad_cfg.get("action_bounds"), action_chunk=grad_cfg.get("action_chunk", 5), process=process)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=plan_cfg, process=process, transform=transform)
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
    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    results_path.mkdir(parents=True, exist_ok=True)
    world.set_policy(policy)
    start = time.time()
    metrics = world.evaluate(dataset=dataset, start_steps=eval_starts.tolist(), goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget, episodes_idx=eval_eps.tolist(), callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True), video=results_path)
    print(metrics)
    with (results_path / cfg.output.filename).open("a") as f:
        f.write(f"\nmetrics: {metrics}\ntime: {time.time()-start}\n")
if __name__ == "__main__":
    run()
