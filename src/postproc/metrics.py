"""DP-aligned decoupled evaluation metrics.

The decoder's four output streams are aligned to the ground truth via
Levenshtein edit distance over composite 4-tuples (substitution cost = 1
iff any of the four fields differ — exactly what tuple equality gives us).
SER, pitch accuracy, and rhythm accuracy are then computed from the
resulting alignment.

Hot path: ``evaluate_ids`` runs once per validation sample (~8,700 samples
per epoch). The alignment uses ``rapidfuzz.distance.Levenshtein.editops``
(C++) which keeps a full validation epoch sub-second. A pure-Python
reference DP is kept as ``_align_python`` for cross-checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rapidfuzz.distance import Levenshtein

from src.data.vocabulary import VocabBundle
from src.postproc.decode import TokenTuple, ids_to_tuples

IdSeqs = tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]
Pair = tuple[TokenTuple | None, TokenTuple | None]


@dataclass(frozen=True)
class AlignmentResult:
    edit_distance: int
    pairs: list[Pair]


@dataclass(frozen=True)
class EvalMetrics:
    ser: float
    pitch_accuracy: float
    rhythm_accuracy: float
    edit_distance: int
    gt_length: int
    pred_length: int
    n_pitch_targets: int
    pitch_correct: int
    n_rhythm_targets: int
    rhythm_correct: int


def _replay_ops(
    gt: Sequence[TokenTuple],
    pred: Sequence[TokenTuple],
    ops,
) -> list[Pair]:
    pairs: list[Pair] = []
    gi = pi = 0
    for op in ops:
        while gi < op.src_pos and pi < op.dest_pos:
            pairs.append((gt[gi], pred[pi]))
            gi += 1
            pi += 1
        tag = op.tag
        if tag == "replace":
            pairs.append((gt[op.src_pos], pred[op.dest_pos]))
            gi = op.src_pos + 1
            pi = op.dest_pos + 1
        elif tag == "delete":
            pairs.append((gt[op.src_pos], None))
            gi = op.src_pos + 1
            pi = op.dest_pos
        elif tag == "insert":
            pairs.append((None, pred[op.dest_pos]))
            gi = op.src_pos
            pi = op.dest_pos + 1
        else:
            raise ValueError(f"unexpected editop tag {tag!r}")
    while gi < len(gt) and pi < len(pred):
        pairs.append((gt[gi], pred[pi]))
        gi += 1
        pi += 1
    return pairs


def align(
    gt: Sequence[TokenTuple],
    pred: Sequence[TokenTuple],
) -> AlignmentResult:
    gt_list = list(gt)
    pred_list = list(pred)
    if gt_list == pred_list:
        return AlignmentResult(
            edit_distance=0,
            pairs=[(g, p) for g, p in zip(gt_list, pred_list)],
        )
    ops = Levenshtein.editops(gt_list, pred_list)
    return AlignmentResult(
        edit_distance=len(ops),
        pairs=_replay_ops(gt_list, pred_list, ops),
    )


def _align_python(
    gt: Sequence[TokenTuple],
    pred: Sequence[TokenTuple],
) -> AlignmentResult:
    """Reference pure-Python DP used only by tests to cross-check ``align``."""
    g = list(gt)
    p = list(pred)
    G, P = len(g), len(p)
    dp = [[0] * (P + 1) for _ in range(G + 1)]
    for i in range(G + 1):
        dp[i][0] = i
    for j in range(P + 1):
        dp[0][j] = j
    for i in range(1, G + 1):
        for j in range(1, P + 1):
            sub = dp[i - 1][j - 1] + (0 if g[i - 1] == p[j - 1] else 1)
            dele = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            dp[i][j] = min(sub, dele, ins)

    pairs: list[Pair] = []
    i, j = G, P
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (0 if g[i - 1] == p[j - 1] else 1)
        ):
            pairs.append((g[i - 1], p[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((g[i - 1], None))
            i -= 1
        else:
            pairs.append((None, p[j - 1]))
            j -= 1
    pairs.reverse()
    return AlignmentResult(edit_distance=dp[G][P], pairs=pairs)


def _counts_from_pairs(pairs: Sequence[Pair]) -> tuple[int, int, int, int]:
    n_pitch = pitch_correct = 0
    n_rhythm = rhythm_correct = 0
    for gt, pred in pairs:
        if gt is None:
            continue
        gt_type = gt[0]
        if gt_type == "note":
            n_pitch += 1
            if pred is not None and pred[1] == gt[1]:
                pitch_correct += 1
        if gt_type in ("note", "rest"):
            n_rhythm += 1
            if pred is not None and pred[2] == gt[2]:
                rhythm_correct += 1
    return n_pitch, pitch_correct, n_rhythm, rhythm_correct


def _safe_div(num: int, den: int) -> float:
    return float("nan") if den == 0 else num / den


def evaluate(
    gt: Sequence[TokenTuple],
    pred: Sequence[TokenTuple],
) -> EvalMetrics:
    res = align(gt, pred)
    gt_len = len(gt)
    pred_len = len(pred)
    n_pitch, pitch_correct, n_rhythm, rhythm_correct = _counts_from_pairs(res.pairs)
    return EvalMetrics(
        ser=res.edit_distance / max(gt_len, 1),
        pitch_accuracy=_safe_div(pitch_correct, n_pitch),
        rhythm_accuracy=_safe_div(rhythm_correct, n_rhythm),
        edit_distance=res.edit_distance,
        gt_length=gt_len,
        pred_length=pred_len,
        n_pitch_targets=n_pitch,
        pitch_correct=pitch_correct,
        n_rhythm_targets=n_rhythm,
        rhythm_correct=rhythm_correct,
    )


def evaluate_ids(
    gt: IdSeqs,
    pred: IdSeqs,
    vocab: VocabBundle,
) -> EvalMetrics:
    gt_tuples = ids_to_tuples(*gt, vocab=vocab, strict=False)
    pred_tuples = ids_to_tuples(*pred, vocab=vocab, strict=False)
    return evaluate(gt_tuples, pred_tuples)


def aggregate(metrics: Sequence[EvalMetrics]) -> EvalMetrics:
    if not metrics:
        return EvalMetrics(
            ser=float("nan"),
            pitch_accuracy=float("nan"),
            rhythm_accuracy=float("nan"),
            edit_distance=0,
            gt_length=0,
            pred_length=0,
            n_pitch_targets=0,
            pitch_correct=0,
            n_rhythm_targets=0,
            rhythm_correct=0,
        )
    edit_distance = sum(m.edit_distance for m in metrics)
    gt_length = sum(m.gt_length for m in metrics)
    pred_length = sum(m.pred_length for m in metrics)
    n_pitch = sum(m.n_pitch_targets for m in metrics)
    pitch_correct = sum(m.pitch_correct for m in metrics)
    n_rhythm = sum(m.n_rhythm_targets for m in metrics)
    rhythm_correct = sum(m.rhythm_correct for m in metrics)
    return EvalMetrics(
        ser=_safe_div(edit_distance, gt_length),
        pitch_accuracy=_safe_div(pitch_correct, n_pitch),
        rhythm_accuracy=_safe_div(rhythm_correct, n_rhythm),
        edit_distance=edit_distance,
        gt_length=gt_length,
        pred_length=pred_length,
        n_pitch_targets=n_pitch,
        pitch_correct=pitch_correct,
        n_rhythm_targets=n_rhythm,
        rhythm_correct=rhythm_correct,
    )


def evaluate_batch(
    gt_batch: Sequence[IdSeqs],
    pred_batch: Sequence[IdSeqs],
    vocab: VocabBundle,
) -> EvalMetrics:
    if len(gt_batch) != len(pred_batch):
        raise ValueError(
            f"gt_batch ({len(gt_batch)}) and pred_batch "
            f"({len(pred_batch)}) length mismatch"
        )
    per_sample = [
        evaluate_ids(gt, pred, vocab) for gt, pred in zip(gt_batch, pred_batch)
    ]
    return aggregate(per_sample)


__all__ = [
    "AlignmentResult",
    "EvalMetrics",
    "aggregate",
    "align",
    "evaluate",
    "evaluate_batch",
    "evaluate_ids",
]
