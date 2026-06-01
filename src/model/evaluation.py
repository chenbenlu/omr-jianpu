from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader

from src.data import VocabBundle
from src.model.model import OMRModel
from src.postproc import EvalMetrics, aggregate, evaluate_batch


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


@torch.no_grad()
def run_ctc_validation(
    model: torch.nn.Module,
    loader: Iterable[dict] | DataLoader,
    vocabs: VocabBundle,
    id_to_joint_token: dict[int, tuple[str, str | None, str | None, str | None]],
) -> EvalMetrics:
    """為 CRNN + CTC 量身打造的驗證循環函數。"""
    model.eval()
    per_batch: list[EvalMetrics] = []

    from src.postproc.ctc_decode import ctc_greedy_decode_batch
    from src.postproc.decode import ids_to_tuples
    from src.postproc.metrics import evaluate

    for batch in loader:
        # 1. 執行 CRNN 前向預測，取得 Argmax IDs 與有效長度遮罩
        preds, lengths = model.predict_greedy(batch["pixel_values"], blank_id=0)

        # 2. 進行 CTC 貪婪解碼（去重、過濾 Blank）還原成 4-Tuple 預測流
        pred_batch_tuples = ctc_greedy_decode_batch(
            preds, lengths, id_to_joint_token, blank_id=0
        )

        # 3. 逐一樣本建立 Ground Truth 4-Tuple 流並與預測流進行 Levenshtein 評估
        B = batch["type_ids"].size(0)
        for i in range(B):
            L = int(batch["label_lengths"][i].item())
            gt_tuples = ids_to_tuples(
                batch["type_ids"][i, :L],
                batch["pitch_ids"][i, :L],
                batch["rhythm_ids"][i, :L],
                batch["attribute_ids"][i, :L],
                vocabs,
                strict=False,
            )
            pred_tuples = pred_batch_tuples[i]
            per_batch.append(evaluate(gt_tuples, pred_tuples))

    return aggregate(per_batch)
