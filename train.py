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


def lejepa_forward1(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

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
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    eqm_lambda = cfg.loss.get("eqm_lambda",0.5)
    eqm_weight = cfg.loss.get("eqm_pred_weight", 1.0)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    
    emb = output["emb"]
    ctx_emb = emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    ctx_actions_raw = batch["action"][:, :ctx_len]
    B = ctx_actions_raw.shape[0]

    with torch.enable_grad():
        gamma = torch.rand(B, 1, 1, device=ctx_actions_raw.device, dtype=ctx_actions_raw.dtype)
        eps = torch.randn_like(ctx_actions_raw)
        act_gamma = (gamma * ctx_actions_raw.detach() + (1 - gamma) * eps).requires_grad_(True)
        pred_emb = self.model.predict(ctx_emb, self.model.action_encoder(act_gamma))

        se_pred = torch.nn.functional.mse_loss(pred_emb, tgt_emb, reduction='none')
        mse_per_sample = se_pred.mean(dim=(1, 2))
        gamma_1d = gamma.view(B)
        output["pred_loss"] = (mse_per_sample * gamma_1d).mean()
    
        energy = se_pred.sum(dim=-1).mean() 
        
        grad_energy = torch.autograd.grad(energy, act_gamma, create_graph=True)[0]
        target_grad = (eps - ctx_actions_raw.detach()) * eqm_lambda * (1 - gamma)
    output["pred_loss_eqm"] = torch.nn.functional.mse_loss(grad_energy, target_grad)

    output["energy"] = energy.detach()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    
    output["loss"] = (
        output["pred_loss"] +
        eqm_weight * output["pred_loss_eqm"] +
        lambd * output["sigreg_loss"]
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    losses_dict[f"{stage}/energy"] = output["energy"]
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
    
def get_latest_checkpoint(run_dir: Path, model_name: str):
    """Find the latest step checkpoint for auto-resume."""
    ckpts = list(run_dir.glob(f"{model_name}_step*.ckpt"))
    if not ckpts:
        return None
    # sort by step number
    def extract_step(p):
        try:
            return int(str(p.stem).split("step=")[-1])
        except:
            return -1
    latest = max(ckpts, key=extract_step)
    print(f"Auto-resuming from: {latest}")
    return latest


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
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
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

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
        'model_opt': {
            "modules": 'model',
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

    ##########################
    ##       training       ##
    ##########################

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
        save_top_k=1,       # only keep the latest to save disk space
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

    # ── Auto-resume from latest step checkpoint if it exists ──
    latest_ckpt = get_latest_checkpoint(run_dir, cfg.output_model_name)

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=latest_ckpt,  # None if no checkpoint exists, auto-resumes if found
    )

    manager()
    return


if __name__ == "__main__":
    run()
