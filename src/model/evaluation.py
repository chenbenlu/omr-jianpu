from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader

from src.data import VocabBundle
from src.postproc import EvalMetrics, aggregate, evaluate_batch
from src.model.model import OMRModel


def _build_pred_batch(
    type_ids: torch.Tensor,
    pitch_ids: torch.Tensor,
    rhythm_ids: torch.Tensor,
    attribute_ids: torch.Tensor,
    lengths: torch.Tensor,
) -> list[tuple[list[int], list[int], list[int], list[int]]]:
    out: list[tuple[list[int], list[int], list[int], list[int]]] = []
    for i in range(type_ids.size(0)):
        L = int(lengths[i].item())
        out.append(
            (
                type_ids[i, :L].tolist(),
                pitch_ids[i, :L].tolist(),
                rhythm_ids[i, :L].tolist(),
                attribute_ids[i, :L].tolist(),
            )
        )
    return out


def _build_gt_batch(
    batch: dict,
) -> list[tuple[list[int], list[int], list[int], list[int]]]:
    lengths = batch["label_lengths"]
    out: list[tuple[list[int], list[int], list[int], list[int]]] = []
    for i in range(batch["type_ids"].size(0)):
        L = int(lengths[i].item())
        out.append(
            (
                batch["type_ids"][i, :L].tolist(),
                batch["pitch_ids"][i, :L].tolist(),
                batch["rhythm_ids"][i, :L].tolist(),
                batch["attribute_ids"][i, :L].tolist(),
            )
        )
    return out


@torch.no_grad()
def run_validation(
    model: OMRModel,
    loader: Iterable[dict] | DataLoader,
    vocabs: VocabBundle,
    max_length: int = 512,
) -> EvalMetrics:
    model.eval()
    per_batch: list[EvalMetrics] = []
    for batch in loader:
        gen = model.generate(batch["pixel_values"], max_length=max_length)
        pred_batch = _build_pred_batch(
            gen.type_ids, gen.pitch_ids, gen.rhythm_ids, gen.attribute_ids, gen.lengths
        )
        gt_batch = _build_gt_batch(batch)
        per_batch.append(evaluate_batch(gt_batch, pred_batch, vocabs))
    return aggregate(per_batch)
