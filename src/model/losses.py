from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data import Vocabulary
from src.model.config import ModelConfig

_STREAMS: tuple[str, ...] = ("type", "pitch", "rhythm", "attribute")
_HEADS_WITH_NULL: tuple[str, ...] = ("pitch", "rhythm", "attribute")


def compute_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    cfg: ModelConfig,
    target_lengths: (
        torch.Tensor | None
    ) = None,  # 新增參數：接收來自 train.py 的真實長度向量
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-head cross-entropy.

    `logits[name]`: (B, L, V_name) — the decoder prepends BOS internally, so
    logits align position-for-position with the raw `labels[name]` (B, L).
    PAD positions are zeroed via `ignore_index`. Optional NULL masking
    (per-head flag) coerces `NULL_ID` to `PAD_ID` in that head's targets only,
    so the head skips NULL positions instead of learning to emit NULL.
    """
    # === feature/B-crnn-and-ctr 修改： 根據模式進行分流 ===
    # == 分流 1：如果開啟 use_ctc 配置，切換至全新的多頭 CTC 損失計算流水線 ==
    if cfg.use_ctc:
        if target_lengths is None:
            raise ValueError(
                "在 CTC 模式下，必須傳入來自運算合約的 target_lengths (batch['label_lengths'])"
            )
        return _compute_ctc_loss(logits, labels, cfg, target_lengths)  # 定義在下面

    # === 分流 2：舊有自迴歸交叉熵損失（保持完全向下相容，Member B 舊實驗絕不崩潰） ===
    losses: dict[str, torch.Tensor] = {}
    for name in _STREAMS:
        head_logits = logits[name]
        targets = labels[name].contiguous()

        if name in _HEADS_WITH_NULL and getattr(cfg.mask_null_in_loss, name):
            targets = torch.where(
                targets == Vocabulary.NULL_ID,
                torch.full_like(targets, Vocabulary.PAD_ID),
                targets,
            )

        weight = None
        if name == "type" and cfg.eos_weight != 1.0:
            weight = torch.ones(
                head_logits.size(-1), device=head_logits.device, dtype=torch.float32
            )
            weight[Vocabulary.EOS_ID] = cfg.eos_weight

        loss = F.cross_entropy(
            head_logits.reshape(-1, head_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=Vocabulary.PAD_ID,
            weight=weight,
        )
        losses[name] = loss

    weights = cfg.loss_weights
    total = (
        weights.type * losses["type"]
        + weights.pitch * losses["pitch"]
        + weights.rhythm * losses["rhythm"]
        + weights.attribute * losses["attribute"]
    )
    return total, losses


def _compute_ctc_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    cfg: ModelConfig,
    target_lengths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """非自迴歸多頭 CTC Loss 的核心實作函數。"""
    losses: dict[str, torch.Tensor] = {}

    for name in _STREAMS:
        head_logits = logits[name]  # 形狀: (B, W, V_name + 1)
        targets = labels[name].contiguous()  # 形狀: (B, L_padded)

        B, W, V_plus_1 = head_logits.shape
        blank_id = V_plus_1 - 1  # 取得步驟二實作的虛擬 Blank ID

        # 1. 影像輸入的時間步長度 (input_lengths)
        # 由於影像經過 CNN 降採樣，RNN 在時間軸上的總長度即為 W
        input_lengths = torch.full((B,), W, dtype=torch.long, device=head_logits.device)

        # 2. 實體標籤的目標長度 (target_lengths)
        # 轉為 PyTorch 規定的長整型，並做 clamp 數值防禦，確保長度大於 0 防止底層 C++ 噴錯
        clamped_target_lengths = torch.clamp(target_lengths.to(torch.long), min=1)

        # 3. 維度與機率域轉換 (Shape & Math Transformation)
        # PyTorch 規定輸入必須是 log_softmax 機率，且形狀排布必須為 (T, B, C)
        # (B, W, V_plus_1) -> log_softmax -> permute 轉置 -> (W, B, V_plus_1)
        log_probs = F.log_softmax(head_logits, dim=-1).permute(1, 0, 2)

        # 4. 調用 PyTorch 核心功能計算單頭 CTC Loss
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=clamped_target_lengths,
            blank=blank_id,
            zero_infinity=True,  # 強健性防禦：若遇到異常樣本導致 Loss 變成無窮大，自動歸零，避免毀掉整個模型梯度
        )

        losses[name] = loss

    # 5. 依據原本配置系統中的分頭損失權重進行綜合加權
    weights = cfg.loss_weights
    total = (
        weights.type * losses["type"]
        + weights.pitch * losses["pitch"]
        + weights.rhythm * losses["rhythm"]
        + weights.attribute * losses["attribute"]
    )

    return total, losses
