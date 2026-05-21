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

    def _compute_energy(
        self,
        ctx_emb: torch.Tensor,   # (1, ctx_len, hidden_dim)  — no grad needed
        ctx_act: torch.Tensor,   # (1, ctx_len, embed_dim)   — no grad needed
        act_seq: torch.Tensor,   # (1, horizon, action_dim)  — requires_grad=True
        goal_emb: torch.Tensor,  # (1, hidden_dim)
    ) -> torch.Tensor:
        """
        Roll out the predictor for `horizon` steps and return the total energy.

        Energy definition matches lejepa_forward exactly:
            E = sum_t  (pred_emb_t - goal_emb).pow(2).sum(dim=-1).mean()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       sum over hidden_dim, then mean over batch

        ARPredictor expects the full context window of action embeddings, so
        we maintain a sliding window that starts with the observed ctx_act and
        is extended with the candidate actions as the rollout proceeds.

        The gradient flows through action_encoder(act_seq) → predict → energy,
        consistent with how grad_energy w.r.t. actions is computed in training.
        """
        # Encode the full candidate sequence through the action encoder so
        # autograd can trace gradients back to act_seq.
        # act_seq: (1, horizon, action_dim) → act_emb_seq: (1, horizon, embed_dim)
        act_emb_seq = self.model.action_encoder(act_seq)

        current_ctx_emb = ctx_emb.detach()   # (1, ctx_len, hidden_dim)
        # Sliding action-embedding window starts with the observed context.
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
            # Matches: energy = (pred - tgt).pow(2).sum(dim=-1).mean()
            total_energy = total_energy + (
                (pred_emb - goal_emb).pow(2).sum(dim=-1).mean()
            )

            # ── slide the observation context window ────────────────────────
            current_ctx_emb = torch.cat(
                [current_ctx_emb[:, 1:], pred_emb.unsqueeze(1).detach()], dim=1
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

    def _preprocess_obs(self, obs: dict) -> dict:
        batch = {}
        for key, val in obs.items():
            if key in self.transform:
                batch[key] = self.transform[key](val).unsqueeze(0)
            elif key in self.process:
                arr = np.array(val, dtype=np.float32).reshape(1, -1)
                batch[key] = torch.tensor(
                    self.process[key].transform(arr), dtype=torch.float32
                )
            else:
                batch[key] = torch.tensor(
                    np.array(val), dtype=torch.float32
                ).unsqueeze(0)
        return batch

    @torch.no_grad()
    def _encode_goal(self, goal_obs: dict) -> torch.Tensor:
        """Encode goal observation → (1, hidden_dim)."""
        batch = self._preprocess_obs(goal_obs)
        device = next(self.solver.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        output = self.solver.model.encode(batch)
        # Take the last frame of the encoded sequence as the goal latent and
        # squeeze to (1, hidden_dim) — consistent with how goal_emb is used in
        # _compute_energy.
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
        if "action" in process:
            action_dim = process["action"].mean_.shape[0]
        else:
            action_dim = getattr(model, "action_dim", 2)
            
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
