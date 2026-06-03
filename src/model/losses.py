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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-head cross-entropy.

    `logits[name]`: (B, L, V_name) — the decoder prepends BOS internally, so
    logits align position-for-position with the raw `labels[name]` (B, L).
    PAD positions are zeroed via `ignore_index`. Optional NULL masking
    (per-head flag) coerces `NULL_ID` to `PAD_ID` in that head's targets only,
    so the head skips NULL positions instead of learning to emit NULL.
    """
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
