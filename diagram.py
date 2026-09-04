import os

# Set EGL for rendering if the model needs to render images internally, 
# though for pure latent planning this might not be strictly necessary.
os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn import preprocessing
import torch
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import stable_worldmodel as swm


def img_transform(cfg):
    """Standard image preprocessing."""
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
    """Loads the HDF5 dataset."""
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


class SingleInstancePreprocessor:
    """
    Handles normalisation and transform for a single data instance
    to prepare it for the World Model.
    """
    def __init__(self, config, process: dict, transform: dict, action_chunk: int):
        self.config = config
        self.process = process
        self.transform = transform
        self.action_chunk = action_chunk

    def preprocess(self, obs: dict) -> dict:
        """Preprocess single obs dict into tensors with leading (B=1, T=1) dims."""
        batch = {}
        for key, val in obs.items():
            if key in self.transform:
                val_np = np.array(val)
                # handle image sequences vs single images
                if val_np.ndim == 4:  # T, C, H, W
                    frames = [self.transform[key](val_np[t]) for t in range(val_np.shape[0])]
                    batch[key] = torch.stack(frames, dim=0).unsqueeze(0)  # 1, T, C, H, W
                else:  # C, H, W
                    batch[key] = self.transform[key](val_np).unsqueeze(0).unsqueeze(0)  # 1, 1, C, H, W

            elif key in self.process:
                # Normalise vector data
                arr = np.array(val, dtype=np.float32).reshape(1, -1)
                transformed = self.process[key].transform(arr)
                t = torch.tensor(transformed, dtype=torch.float32)
                
                if key == "action":
                    # Action encoder expects chunked inputs
                    tiled = t.repeat(1, self.action_chunk)
                    batch[key] = tiled.unsqueeze(1)  # 1, 1, D*chunk
                else:
                    batch[key] = t.unsqueeze(1)  # 1, 1, D
            else:
                # Pass through other keys (like step_idx) if needed, adding dimensions
                if isinstance(val, (np.ndarray, torch.Tensor)):
                    t = torch.as_tensor(val, dtype=torch.float32)
                    # Add B, T dims based on heuristic
                    while t.ndim < 2:
                        t = t.unsqueeze(0)
                    batch[key] = t
                else:
                    continue
        return batch


def run_optimization_and_plot(
    model, 
    preprocessor, 
    context_data, 
    goal_data, 
    plan_cfg, 
    grad_cfg, 
    device
):
    """
    Runs the latent planning optimization loop for a single instance
    using explicit energy gradient calculation (torch.autograd.grad).
    """
    print("\nPreparing optimization instance...")
    
    # 1. Preprocess and move to device
    proc_context = preprocessor.preprocess(context_data)
    proc_goal = preprocessor.preprocess(goal_data)
    
    proc_context = {k: v.to(device) for k, v in proc_context.items()}
    proc_goal = {k: v.to(device) for k, v in proc_goal.items()}

    # 2. Encode context and goal
    with torch.no_grad():
        ctx_output = model.encode(proc_context)
        ctx_emb = ctx_output["emb"]        # (1, T_ctx, hidden_dim)
        ctx_act = ctx_output["act_emb"]    # (1, T_ctx, embed_dim)

        goal_output = model.encode(proc_goal)
        goal_emb = goal_output["emb"][:, -1]  # (1, hidden_dim)

    # 3. Setup Optimization variables
    horizon = plan_cfg.horizon
    action_dim = preprocessor.process["action"].mean_.shape[0]
    chunk_dim = preprocessor.action_chunk * action_dim
    
    # Initialize random action sequence (B=1)
    act_seq = torch.randn(
        1, horizon, chunk_dim, device=device, requires_grad=True
    )
    
    # Apply bounds if configured
    if grad_cfg.get("action_bounds") and "action" in preprocessor.process:
        raw_lo, raw_hi = grad_cfg.action_bounds
        proc = preprocessor.process["action"]
        norm_lo = proc.transform(np.array([[raw_lo] * action_dim]))[0, 0]
        norm_hi = proc.transform(np.array([[raw_hi] * action_dim]))[0, 0]
        with torch.no_grad():
            act_seq.clamp_(norm_lo, norm_hi)

    lr = grad_cfg.get("lr", 0.1)
    n_iter = grad_cfg.get("n_iter", 50)
    mse_history = []

    print(f"Starting gradient optimization for {n_iter} iterations (lr={lr})...")
    start_time = time.time()

    # 4. Optimization Loop
    for i in range(n_iter):
        # Forward Rollout in Latent Space
        act_emb_seq = model.action_encoder(act_seq)  # (1, horizon, embed_dim)

        current_ctx_emb = ctx_emb
        current_ctx_act = ctx_act
        final_pred_emb = None

        for t in range(horizon):
            step_act_emb = act_emb_seq[:, t : t + 1]
            # Shift context window: remove oldest, add current action
            full_act_ctx = torch.cat(
                [current_ctx_act[:, 1:], step_act_emb], dim=1
            )

            # Predict next state embedding
            pred_out = model.predict(current_ctx_emb, full_act_ctx)
            pred_emb = pred_out[:, -1] if pred_out.dim() == 3 else pred_out 

            # Update context for next step: remove oldest emb, add predicted emb
            current_ctx_emb = torch.cat(
                [current_ctx_emb[:, 1:], pred_emb.unsqueeze(1)], dim=1
            )
            current_ctx_act = full_act_ctx
            final_pred_emb = pred_emb

        # --- Compute Energy and Autograd Gradient ---
        # energy = (pred - target)^2 summed over latent dimensions and batch
        energy = (final_pred_emb - goal_emb.detach()).pow(2).mean()
        
        # Calculate exact gradient wrt act_seq
        grad_energy = torch.autograd.grad(
            energy, 
            act_seq, 
            create_graph=False
        )[0]

        # Record history
        mse_history.append(energy.item())

        # Manual Gradient Step (Gradient Descent on Energy)
        with torch.no_grad():
            # Grad clipping
            if grad_cfg.get("grad_clip"):
                grad_norm = grad_energy.norm()
                if grad_norm > grad_cfg.grad_clip:
                    grad_energy = grad_energy * (grad_cfg.grad_clip / (grad_norm + 1e-6))

            # Update: act = act - lr * grad_energy
            act_seq -= lr * grad_energy

            # Add Langevin dynamics noise (if configured)
            if grad_cfg.get("action_noise", 0.0) > 0.0:
                act_seq += grad_cfg.action_noise * torch.randn_like(act_seq)

            # Project back to bounds
            if grad_cfg.get("action_bounds"):
                act_seq.clamp_(norm_lo, norm_hi)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iteration {i+1:3d}/{n_iter}  Energy (MSE): {mse_history[-1]:.6f}")

    total_time = time.time() - start_time
    print(f"Optimization finished in {total_time:.2f}s.")

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_iter + 1), mse_history, marker='.', linestyle='-', color='b')
    plt.title(f'Planning Optimization: Energy Minimization\n(Horizon: {horizon})')
    plt.xlabel('Gradient Descent Iteration')
    plt.ylabel('Energy / Latent MSE')
    plt.grid(True, which="both", ls="-", alpha=0.5)

    output_filename = "plan_optimization_curve.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\nPlot saved to: {output_filename}")
    plt.show()


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    policy_name = cfg.get("policy", "random")
    if policy_name == "random":
        raise ValueError("Cannot run gradient optimization visualization for 'random' policy. "
                         "Please specify a trained policy in config (e.g., +policy=jepa).")
        
    ckpt_path = cfg.policy + "_object.ckpt"
    print(f"Loading model object from {ckpt_path}...")
    # Load weights_only=False because we are loading the whole object, not just state_dict
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)  # Weights are frozen
    model.interpolate_pos_encoding = True

    # 2. Load Dataset & Statistics
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    
    # Build StandardScalers for normalization based on full dataset
    process_meta = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue  # Images handled via torchvision transform
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        # Remove NaNs for fitting stats
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process_meta[col] = processor

    # 3. Determine Model Hyperparameters (Context length, chunking)
    raw_action_dim = process_meta["action"].mean_.shape[0]
    
    # Determine Action Chunk Size
    if hasattr(cfg, "wm") and hasattr(cfg.wm, "action_chunk"):
        action_chunk = cfg.wm.action_chunk
    elif hasattr(model, "action_encoder") and hasattr(model.action_encoder, "in_features"):
        action_chunk = model.action_encoder.in_features // raw_action_dim
    else:
        action_chunk = cfg.get("gradient_solver", {}).get("action_chunk", 5)

    # Determine Context Length
    if hasattr(cfg, "wm") and hasattr(cfg.wm, "history_size"):
        ctx_len = cfg.wm.history_size
    elif hasattr(cfg, "model") and hasattr(cfg.model, "history_size"):
        ctx_len = cfg.model.history_size
    else:
        ctx_len = getattr(model, "history_size", 1)

    # Initialize Preprocessor
    transforms_meta = {
        "pixels": img_transform(cfg),
        "goal":   img_transform(cfg),  # typically just standard image transform
    }
    preprocessor = SingleInstancePreprocessor(cfg, process_meta, transforms_meta, action_chunk)

    # 4. Extract ONE valid sample instance from dataset
    print("\nExtracting sample data from dataset...")
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    
    sample_idx = 0 
    row_data = dataset.get_row_data(sample_idx)
    ep_id = row_data[col_name]
    start_step_idx = row_data["step_idx"]

    # Context history extraction
    indices = np.arange(max(0, sample_idx - ctx_len + 1), sample_idx + 1)
    context_rows = dataset.get_row_data(indices)
    actual_ctx_len = len(indices)
    
    context_data = {}
    for key in context_rows.keys():
        if key in ["episode_idx", "ep_idx", "step_idx"]: 
            continue
            
        data = context_rows[key]
        if actual_ctx_len < ctx_len:
            pad_len = ctx_len - actual_ctx_len
            padding = np.repeat(data[0:1], pad_len, axis=0)
            data = np.concatenate([padding, data], axis=0)
        context_data[key] = data

    # Extract Goal
    goal_offset = cfg.eval.goal_offset_steps
    ep_episode_idx_all = dataset.get_col_data(col_name)
    ep_mask = ep_episode_idx_all == ep_id
    ep_step_idx_all = dataset.get_col_data("step_idx")
    
    max_step_in_ep = np.max(ep_step_idx_all[ep_mask])
    goal_step_idx = min(start_step_idx + goal_offset, max_step_in_ep)
    
    goal_abs_idx_mask = ep_mask & (ep_step_idx_all == goal_step_idx)
    goal_abs_idx = np.nonzero(goal_abs_idx_mask)[0][0]
    goal_row = dataset.get_row_data(goal_abs_idx)
    
    goal_data = {}
    goal_data["pixels"] = goal_row["pixels"] 
    if "agent_pos" in goal_row:
        goal_data["agent_pos"] = goal_row["agent_pos"]

    print(f"Successfully extracted instance from Ep {ep_id}. "
          f"Start Step: {start_step_idx}, Goal Step: {goal_step_idx} "
          f"(Offset: {goal_step_idx - start_step_idx})")

    # 5. Run Optimization and Generate Plot
    plan_config = swm.PlanConfig(**cfg.plan_config)
    grad_config = cfg.get("gradient_solver", {})
    
    grad_config["n_iter"] = 500

    run_optimization_and_plot(
        model,
        preprocessor,
        context_data,
        goal_data,
        plan_config,
        grad_config,
        device
    )


if __name__ == "__main__":
    main()
