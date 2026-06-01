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
    # ==========================================
    # ==== 修改後程式碼：宣告並將變數綁定至全域範疇
    # ==========================================
    # --- 新增：定義並初始化 CTC 聯合詞表全域變數 ---
    joint_token_to_id = None
    id_to_joint_token = None
    if cfg.model.get("is_crnn", False):
        from src.data.vocabulary import build_joint_ctc_vocab

        joint_token_to_id, id_to_joint_token = build_joint_ctc_vocab(vb)
        # 動態將詞表真實大小覆寫回 Hydra 配置中，確保與外部定義完美同步
        cfg.model.ctc_vocab_size = len(id_to_joint_token)

    spec = get_encoder_spec(cfg.model.encoder_name)
    encoder = hydra.utils.instantiate(cfg.model.encoder, encoder_spec=spec)
    model_cfg = _build_model_config(cfg)

    # ==========================================
    # ==== 修改後程式碼：依組態實例化 OMRCRNNModel
    # ==========================================
    # --- 新增：動態切換 CRNN 模型與一體化詞表初始化 ---
    if cfg.model.get("is_crnn", False):
        from src.model.crnn import OMRCRNNModel

        model = OMRCRNNModel(
            encoder=encoder,
            ctc_vocab_size=cfg.model.ctc_vocab_size,
            d_model=cfg.model.d_model,
        )
        # 此處需為訓練流程配上全域載入的 joint_token_to_id 字典
    else:
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
            spec.build_train_transform(),
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
                # ==========================================
                # ==== 修改後程式碼：加入 CTC 損失函數與流打包計算
                # ==========================================
                # --- 新增：CRNN CTC 損失流計算 ---
                if cfg.model.get("is_crnn", False):
                    from src.model.ctc_losses import (
                        compute_ctc_loss,
                        pack_streams_to_ctc_targets,
                    )

                    # 1. 前向傳播計算序列 Logits
                    out = model(pixel_values=batch["pixel_values"])

                    # 2. 將四大獨立標籤流打包為一體化 CTC Targets
                    ctc_targets, target_lengths = pack_streams_to_ctc_targets(
                        batch, vb, joint_token_to_id
                    )

                    # 3. 計算 CTC 損失
                    total = compute_ctc_loss(
                        logits=out["logits"],
                        ctc_targets=ctc_targets,
                        attention_mask=out["attention_mask"],
                        target_lengths=target_lengths,
                        blank_id=0,
                    )
                    # 構造虛擬字典以便 TensorBoard 日誌模組能直接複用
                    per_head = {
                        "type": total,
                        "pitch": total,
                        "rhythm": total,
                        "attribute": total,
                    }
                else:
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
                # ==========================================
                # ==== 修改後程式碼：路由切換至 CTC 的特殊驗證函數
                # ==========================================
                if cfg.model.get("is_crnn", False):
                    # 呼叫為 CRNN 量身訂製的 Validation 腳本
                    # (內部將 model.predict_greedy 產生的結果透過 ctc_greedy_decode_batch 轉回 tuples)
                    from src.model.evaluation import run_ctc_validation

                    metrics = run_ctc_validation(
                        accelerator.unwrap_model(model),
                        val_loader,
                        vb,
                        id_to_joint_token,
                    )
                else:
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
                    # ==========================================
                    # ==== 修改後程式碼：自動將詞表與模型权重複製綁定
                    # ==========================================
                    save_path = ckpt_root / f"step-{step}-best"
                    accelerator.save_state(str(save_path))
                    # --- 新增：如果是 CRNN 模式，將解碼用 JSON 一併存入該 Step 目錄 ---
                    if (
                        cfg.model.get("is_crnn", False)
                        and id_to_joint_token is not None
                    ):
                        import json

                        vocab_file = save_path / "ctc_vocab.json"
                        with vocab_file.open("w", encoding="utf-8") as f:
                            # 由於 JSON 的 key 必須是字串，轉換 Tuple 為 List 儲存
                            payload = {
                                "id_to_joint_token": {
                                    k: list(v) for k, v in id_to_joint_token.items()
                                }
                            }
                            json.dump(payload, f, ensure_ascii=False, indent=2)
                model.train()

    if accelerator.is_local_main_process:
        logger.close()


if __name__ == "__main__":
    main()
