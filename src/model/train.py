from __future__ import annotations

import re
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from src.data import build_default_vocabs, create_dataloaders, get_encoder_spec
from src.model.config import LossWeights, MaskNullInLoss, ModelConfig
from src.model.evaluation import run_validation
from src.model.losses import compute_loss
from src.model.model import OMRModel

_STEP_BEST_RE = re.compile(r"step-(\d+)-best")


def _resolve_checkpoint(ckpt_dir: Path) -> Path:
    """Return the directory holding `model.safetensors`.

    Accepts either a leaf checkpoint dir or a run dir containing many
    `step-N-best/` subdirs; in the latter case picks the highest step (the best
    val-SER snapshot, saved last). Mirrors the deploy-side resolver but is kept
    local so `src.model` does not import `src.deploy`.
    """
    if (ckpt_dir / "model.safetensors").exists():
        return ckpt_dir
    candidates = [
        (int(m.group(1)), p)
        for p in ckpt_dir.iterdir()
        if p.is_dir() and (m := _STEP_BEST_RE.fullmatch(p.name))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no model.safetensors and no step-N-best/ subdir under {ckpt_dir}"
        )
    return max(candidates, key=lambda c: c[0])[1]


def _save_model_only(accelerator: Accelerator, model: OMRModel, out_dir: Path) -> None:
    """Write just `model.safetensors` (no optimizer/rng) for a periodic snapshot.

    Lets fine-tuning keep multiple selectable candidates cheaply (~0.4 GB each
    vs ~1.1 GB for a full `save_state`). The dir is named `step-N` (no `-best`)
    so the best-checkpoint resolver ignores it; point inference at it directly.
    """
    if not accelerator.is_local_main_process:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = accelerator.unwrap_model(model).state_dict()
    sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
    save_file(sd, str(out_dir / "model.safetensors"))


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

    # Warm-start (fine-tuning): load weights from a prior checkpoint before the
    # optimizer/accelerator wrap them. strict=False tolerates head-naming drift;
    # missing/unexpected should both be 0 for an identical architecture.
    init_from = cfg.checkpoint.get("init_from")
    if init_from:
        wdir = _resolve_checkpoint(Path(init_from))
        missing, unexpected = model.load_state_dict(
            load_file(str(wdir / "model.safetensors")), strict=False
        )
        accelerator.print(
            f"warm-start from {wdir}: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    aug_profile = cfg.data.get("aug_profile", "default")
    loaders = create_dataloaders(
        out_dir=cfg.data.out_dir,
        encoder=spec,
        train_size=cfg.data.train_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.seed,
        aug_profile=aug_profile,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    # If a pre-rendered train dir is given, read PNGs from disk instead of
    # rendering with verovio every epoch (the on-the-fly path is CPU-bound and
    # starves the GPU). Augmentation is still applied at load time, so the
    # training distribution is unchanged.
    train_dir = cfg.data.get("train_dir")
    if train_dir:
        from torch.utils.data import DataLoader

        from src.data import PreRenderedOMRDataset, collate_fn

        train_ds = PreRenderedOMRDataset(
            Path(train_dir) / "manifest.jsonl",
            vb,
            spec.build_train_transform(aug_profile),
            max_seq_len=cfg.data.max_seq_len,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=cfg.data.num_workers > 0,
            collate_fn=collate_fn,
        )

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

            save_every = cfg.train.get("save_every_steps", 0)
            if (
                save_every
                and step % save_every == 0
                and accelerator.is_local_main_process
            ):
                _save_model_only(accelerator, model, ckpt_root / f"step-{step}")

    if accelerator.is_local_main_process:
        logger.close()


if __name__ == "__main__":
    main()
