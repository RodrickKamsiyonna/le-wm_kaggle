import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


# ─────────────────────────────────────────────────────────────────────────────
# Resumable DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class ResumableDataLoader(torch.utils.data.DataLoader):
    """DataLoader that supports state_dict / load_state_dict for mid-epoch resume."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_idx = 0

    def state_dict(self):
        return {"start_idx": self._start_idx}

    def load_state_dict(self, state: dict):
        self._start_idx = state.get("start_idx", 0)

    def __iter__(self):
        iterator = super().__iter__()
        for i, batch in enumerate(iterator):
            if i < self._start_idx:
                continue
            self._start_idx = 0
            yield batch
        self._start_idx = 0


# ─────────────────────────────────────────────────────────────────────────────
# Forward passes
# ─────────────────────────────────────────────────────────────────────────────

def lejepa_forward1(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses (baseline)."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses with EQM."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    eqm_lambda = cfg.loss.get("eqm_lambda", 1.0)
    eqm_weight = cfg.loss.get("eqm_pred_weight", 0.5)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]

    pred_emb_std = self.model.predict(ctx_emb, ctx_act)
    output["pred_loss"] = (pred_emb_std - tgt_emb).pow(2).mean()

    ctx_actions_raw = batch["action"][:, :ctx_len]
    B = ctx_actions_raw.shape[0]

    with torch.enable_grad():
        gamma = torch.rand(B, 1, 1, device=ctx_actions_raw.device, dtype=ctx_actions_raw.dtype)
        eps = torch.randn_like(ctx_actions_raw)

        act_gamma = (
            gamma * ctx_actions_raw.detach() + (1 - gamma) * eps
        ).requires_grad_(True)

        pred_emb_noisy = self.model.predict(
            ctx_emb.detach(),
            self.model.action_encoder(act_gamma),
        )

        energy = (pred_emb_noisy - tgt_emb.detach()).pow(2).sum(dim=-1).mean()
        grad_energy = torch.autograd.grad(energy, act_gamma, create_graph=True)[0]
        target_grad = (eps - ctx_actions_raw.detach()) * eqm_lambda * (1 - gamma)

    output["pred_loss_eqm"] = (grad_energy - target_grad).pow(2).mean()
    output["energy"] = energy.detach()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    output["loss"] = (
        output["pred_loss"]
        + eqm_weight * output["pred_loss_eqm"]
        + lambd * output["sigreg_loss"]
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    losses_dict[f"{stage}/energy"] = output["energy"]
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_checkpoint(run_dir: Path, model_name: str):
    """Find the latest step checkpoint for auto-resume."""
    ckpts = list(run_dir.glob(f"{model_name}_step*.ckpt"))
    if not ckpts:
        return None

    def extract_step(p):
        try:
            return int(str(p.stem).split("step=")[-1])
        except Exception:
            return -1

    latest = max(ckpts, key=extract_step)
    print(f"Auto-resuming from: {latest}")
    return latest


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    
    # --- BYPASS LIBRARY RESOLUTION BUG ---
    # Construct the absolute path based on where we know the file is in Kaggle
    absolute_path = f"/kaggle/working/stablewm/datasets/{dataset_name}.h5"
    
    if os.path.exists(absolute_path):
        print(f"Bypassing resolver. Loading explicit path: {absolute_path}")
        dataset_target = absolute_path
    else:
        # Fallback just in case
        dataset_target = dataset_name
    # -------------------------------------

    dataset = swm.data.load_dataset(
        dataset_target, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        # We ensure wm dictionary exists to avoid AttributeError if config structure changed
        if not hasattr(cfg, "wm"):
            cfg.wm = {}
            
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = ResumableDataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = ResumableDataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    # ── model / optimiser ─────────────────────────────────────────────────────
    
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    
    # Safer action dimension extraction (prevents crash if 'action' isn't in keys_to_load)
    action_dim = getattr(cfg.wm, "action_dim", dataset.get_dim("action"))
    effective_act_dim = cfg.data.dataset.frameskip * action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    steps_per_epoch = len(train)
    total_steps = cfg.trainer.max_epochs * steps_per_epoch
    warmup_steps = int(0.03 * total_steps)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": total_steps,
            },
            "interval": "step",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    # ── training ──────────────────────────────────────────────────────────────
    run_dir = Path("/kaggle/working", cfg.get("subdir") or "lewm_run")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    step_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename=f"{cfg.output_model_name}_step{{step}}",
        every_n_train_steps=500,
        save_top_k=1,
        save_last=False,
    )

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[step_checkpoint, object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    latest_ckpt = get_latest_checkpoint(run_dir, cfg.output_model_name)

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=latest_ckpt,
    )

    manager()


if __name__ == "__main__":
    run()
