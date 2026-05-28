from __future__ import annotations

from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import build_default_vocabs, create_dataloaders, get_encoder_spec
from src.model.config import LossWeights, MaskNullInLoss, ModelConfig
from src.model.evaluation import run_validation
from src.model.losses import compute_loss
from src.model.model import OMRModel


def _build_model_config(cfg: DictConfig) -> ModelConfig:
    return ModelConfig(
        d_model=cfg.model.d_model,
        decoder_layers=cfg.model.decoder_layers,
        decoder_heads=cfg.model.decoder_heads,
        decoder_ffn_dim=cfg.model.decoder_ffn_dim,
        dropout=cfg.model.dropout,
        max_decoder_positions=cfg.model.max_decoder_positions,
        scale_embedding=cfg.model.get("scale_embedding", True),
        eos_weight=cfg.model.get("eos_weight", 1.0),
        loss_weights=LossWeights(**OmegaConf.to_container(cfg.model.loss_weights)),
        mask_null_in_loss=MaskNullInLoss(
            **OmegaConf.to_container(cfg.model.mask_null_in_loss)
        ),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    torch.manual_seed(cfg.seed)

    vb = build_default_vocabs()
    spec = get_encoder_spec(cfg.model.encoder_name)
    encoder = hydra.utils.instantiate(cfg.model.encoder, encoder_spec=spec)
    model_cfg = _build_model_config(cfg)
    model = OMRModel(encoder=encoder, vocabs=vb, cfg=model_cfg)

    loaders = create_dataloaders(
        out_dir=cfg.data.out_dir,
        encoder=spec,
        train_size=cfg.data.train_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.seed,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    optimizer = hydra.utils.instantiate(cfg.optim.optimizer, params=model.parameters())
    total_steps = cfg.train.epochs * len(train_loader)
    scheduler = hydra.utils.instantiate(
        cfg.optim.scheduler, optimizer=optimizer, num_training_steps=total_steps
    )
    logger = hydra.utils.instantiate(cfg.logging.logger, run_name=cfg.run_name)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    ckpt_root = Path(cfg.checkpoint.dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_ser = float("inf")
    step = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            disable=not accelerator.is_local_main_process,
        ):
            with accelerator.autocast():
                out = model(
                    pixel_values=batch["pixel_values"],
                    type_ids=batch["type_ids"],
                    pitch_ids=batch["pitch_ids"],
                    rhythm_ids=batch["rhythm_ids"],
                    attribute_ids=batch["attribute_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                )
                labels = {
                    "type": batch["type_ids"],
                    "pitch": batch["pitch_ids"],
                    "rhythm": batch["rhythm_ids"],
                    "attribute": batch["attribute_ids"],
                }
                total, per_head = compute_loss(out["logits"], labels, model_cfg)

            accelerator.backward(total)
            accelerator.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if (
                step % cfg.train.log_every_steps == 0
                and accelerator.is_local_main_process
            ):
                logger.log_scalars(
                    {
                        "train/loss/total": float(total.detach().item()),
                        "train/loss/type": float(per_head["type"].detach().item()),
                        "train/loss/pitch": float(per_head["pitch"].detach().item()),
                        "train/loss/rhythm": float(per_head["rhythm"].detach().item()),
                        "train/loss/attribute": float(
                            per_head["attribute"].detach().item()
                        ),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step,
                )

            if (
                step % cfg.train.eval_every_steps == 0
                and accelerator.is_local_main_process
            ):
                metrics = run_validation(
                    accelerator.unwrap_model(model),
                    val_loader,
                    vb,
                    max_length=cfg.train.gen_max_length,
                )
                logger.log_scalars(
                    {
                        "val/ser": metrics.ser,
                        "val/pitch_accuracy": metrics.pitch_accuracy,
                        "val/rhythm_accuracy": metrics.rhythm_accuracy,
                    },
                    step,
                )
                if metrics.ser < best_ser:
                    best_ser = metrics.ser
                    accelerator.save_state(str(ckpt_root / f"step-{step}-best"))
                model.train()

    if accelerator.is_local_main_process:
        logger.close()


if __name__ == "__main__":
    main()
