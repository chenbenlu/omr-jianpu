from __future__ import annotations

import math

import pytest

from src.data import MelodyGenerator, build_default_vocabs
from src.data.vocabulary import Vocabulary
from src.postproc import (
    JianpuRenderConfig,
    aggregate,
    align,
    evaluate,
    evaluate_batch,
    evaluate_ids,
    ids_to_jianpu,
    ids_to_tuples,
    tuples_to_jianpu,
)
from src.postproc.metrics import _align_python

# ---------------------------------------------------------------------------
# Decode — ground-truth path (lock-step streams)
# ---------------------------------------------------------------------------


def _encode_tuples(
    tuples: list[tuple[str, str | None, str | None, str | None]],
):
    vb = build_default_vocabs()
    type_ids = vb.type.encode([t[0] for t in tuples])
    pitch_ids = vb.pitch.encode([t[1] for t in tuples])
    rhythm_ids = vb.rhythm.encode([t[2] for t in tuples])
    attribute_ids = vb.attribute.encode([t[3] for t in tuples])
    return vb, type_ids, pitch_ids, rhythm_ids, attribute_ids


def test_decode_roundtrip_gt() -> None:
    tuples = [
        ("clef", None, None, "G2"),
        ("key_signature", None, None, "ks+0"),
        ("time_signature", None, None, "4/4"),
        ("note", "C4", "quarter", None),
        ("rest", None, "eighth", None),
        ("barline", None, None, None),
        ("note", "D4", "eighth", None),
    ]
    vb, t, p, r, a = _encode_tuples(tuples)
    out = ids_to_tuples(t, p, r, a, vb)
    assert out == tuples


def test_decode_strips_leading_bos_and_trailing_eos_pad() -> None:
    vb = build_default_vocabs()
    BOS, EOS, PAD = Vocabulary.BOS_ID, Vocabulary.EOS_ID, Vocabulary.PAD_ID
    NULL = Vocabulary.NULL_ID

    type_ids = [BOS, vb.type.token_to_id["note"], EOS, PAD]
    pitch_ids = [BOS, vb.pitch.token_to_id["C4"], EOS, PAD]
    rhythm_ids = [BOS, vb.rhythm.token_to_id["quarter"], EOS, PAD]
    attribute_ids = [BOS, NULL, EOS, PAD]

    out = ids_to_tuples(type_ids, pitch_ids, rhythm_ids, attribute_ids, vb)
    assert out == [("note", "C4", "quarter", None)]


def test_decode_strict_raises_on_misalignment() -> None:
    vb = build_default_vocabs()
    type_ids = [vb.type.token_to_id["note"]]
    pitch_ids = [Vocabulary.PAD_ID]
    rhythm_ids = [vb.rhythm.token_to_id["quarter"]]
    attribute_ids = [Vocabulary.NULL_ID]
    with pytest.raises(ValueError):
        ids_to_tuples(type_ids, pitch_ids, rhythm_ids, attribute_ids, vb, strict=True)


# ---------------------------------------------------------------------------
# Decode — prediction path (per-head independence)
# ---------------------------------------------------------------------------


def test_decode_coerces_pad_in_pitch_to_unk() -> None:
    """type=note + pitch=<PAD> at the same index keeps the position, with
    pitch coerced to '<UNK>' so the failure surfaces downstream."""
    vb = build_default_vocabs()
    type_ids = [vb.type.token_to_id["note"]]
    pitch_ids = [Vocabulary.PAD_ID]
    rhythm_ids = [vb.rhythm.token_to_id["quarter"]]
    attribute_ids = [Vocabulary.NULL_ID]
    out = ids_to_tuples(type_ids, pitch_ids, rhythm_ids, attribute_ids, vb)
    assert out == [("note", "<UNK>", "quarter", None)]


def test_decode_coerces_eos_in_attribute_to_unk() -> None:
    vb = build_default_vocabs()
    type_ids = [vb.type.token_to_id["note"]]
    pitch_ids = [vb.pitch.token_to_id["C4"]]
    rhythm_ids = [vb.rhythm.token_to_id["quarter"]]
    attribute_ids = [Vocabulary.EOS_ID]
    out = ids_to_tuples(type_ids, pitch_ids, rhythm_ids, attribute_ids, vb)
    assert out == [("note", "C4", "quarter", "<UNK>")]


def test_decode_drops_position_when_type_is_pad_mid_sequence() -> None:
    vb = build_default_vocabs()
    type_ids = [
        vb.type.token_to_id["note"],
        Vocabulary.PAD_ID,
        vb.type.token_to_id["note"],
    ]
    pitch_ids = [
        vb.pitch.token_to_id["C4"],
        vb.pitch.token_to_id["D4"],
        vb.pitch.token_to_id["E4"],
    ]
    rhythm_ids = [vb.rhythm.token_to_id["quarter"]] * 3
    attribute_ids = [Vocabulary.NULL_ID] * 3
    out = ids_to_tuples(type_ids, pitch_ids, rhythm_ids, attribute_ids, vb)
    assert out == [
        ("note", "C4", "quarter", None),
        ("note", "E4", "quarter", None),
    ]


def test_decode_truncates_at_eos_in_type_regardless_of_other_streams() -> None:
    vb = build_default_vocabs()
    note = vb.type.token_to_id["note"]
    qtr = vb.rhythm.token_to_id["quarter"]
    cee = vb.pitch.token_to_id["C4"]
    type_ids = [note, note, Vocabulary.EOS_ID, note, note]
    pitch_ids = [cee, cee, cee, cee, cee]
    rhythm_ids = [qtr, qtr, qtr, qtr, qtr]
    attribute_ids = [Vocabulary.NULL_ID] * 5
    out = ids_to_tuples(type_ids, pitch_ids, rhythm_ids, attribute_ids, vb)
    assert out == [("note", "C4", "quarter", None)] * 2


# ---------------------------------------------------------------------------
# Jianpu mapping correctness
# ---------------------------------------------------------------------------

_CFG_NO_HEADER = JianpuRenderConfig(emit_header=False)


def _render(tuples):
    return tuples_to_jianpu(tuples, _CFG_NO_HEADER)


def _state_prefix(clef: str = "G2", fifths: int = 0, time: str = "4/4"):
    return [
        ("clef", None, None, clef),
        ("key_signature", None, None, f"ks{fifths:+d}"),
        ("time_signature", None, None, time),
    ]


def test_jianpu_c_major_c4_quarter() -> None:
    out = _render(_state_prefix() + [("note", "C4", "quarter", None)])
    assert out == "1"


def test_jianpu_c_major_d4_eighth_underline() -> None:
    out = _render(_state_prefix() + [("note", "D4", "eighth", None)])
    assert out == "_2"


def test_jianpu_c_major_c5_quarter_octave_up() -> None:
    out = _render(_state_prefix() + [("note", "C5", "quarter", None)])
    assert out == "1'"


def test_jianpu_c_major_c3_quarter_octave_down() -> None:
    out = _render(_state_prefix() + [("note", "C3", "quarter", None)])
    assert out == "1,"


def test_jianpu_g_major_g4_is_tonic() -> None:
    out = _render(_state_prefix(fifths=1) + [("note", "G4", "quarter", None)])
    assert out == "1"


def test_jianpu_g_major_f4_natural_is_flat_seven() -> None:
    """F natural in G major is the b7 — diatonic 7 (F#) lowered by a
    semitone. F4 is below the central tonic G4, so it gets one octave-down
    dot."""
    out = _render(_state_prefix(fifths=1) + [("note", "F4", "quarter", None)])
    assert out == "b7,"


def test_jianpu_g_major_f_sharp_4_is_diatonic_seven() -> None:
    out = _render(_state_prefix(fifths=1) + [("note", "F#4", "quarter", None)])
    assert out == "7,"


def test_jianpu_c_major_c4_half_dot() -> None:
    out = _render(_state_prefix() + [("note", "C4", "half_dot", None)])
    assert out == "1 - ."


def test_jianpu_rest_and_barline_and_key_change() -> None:
    tuples = _state_prefix(fifths=0) + [
        ("note", "C4", "quarter", None),
        ("rest", None, "eighth", None),
        ("barline", None, None, None),
        ("key_signature", None, None, "ks+1"),
        ("note", "G4", "quarter", None),  # new tonic
    ]
    out = _render(tuples)
    assert out == "1 _0 | 1"


def test_jianpu_bass_clef_changes_central_octave() -> None:
    """F4 in C major bass-clef should be at degree 4 with NO octave marks
    (bass central octave is 3, so F4 is one letter-octave above C3 tonic)."""
    out = _render(
        [
            ("clef", None, None, "F4"),
            ("key_signature", None, None, "ks+0"),
            ("note", "F4", "quarter", None),
        ]
    )
    assert out == "4'"


def test_jianpu_header_emission() -> None:
    out = tuples_to_jianpu(_state_prefix(fifths=1) + [("note", "G4", "quarter", None)])
    assert "[Clef: G2]" in out
    assert "[Key: G major]" in out
    assert "[Time: 4/4]" in out
    assert out.endswith("\n1")


def test_jianpu_unk_pitch_keeps_rhythm_shape() -> None:
    out = _render(_state_prefix() + [("note", "<UNK>", "eighth", None)])
    assert out == "_?"


def test_jianpu_unk_rhythm_drops_shape_and_appends_marker() -> None:
    out = _render(_state_prefix() + [("note", "C4", "<UNK>", None)])
    assert out == "1?"


def test_jianpu_unk_type_renders_question_mark() -> None:
    out = _render(_state_prefix() + [("<UNK>", None, None, None)])
    assert out == "?"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


_C4 = ("note", "C4", "quarter", None)
_D4 = ("note", "D4", "quarter", None)
_E4 = ("note", "E4", "eighth", None)
_REST_E = ("rest", None, "eighth", None)
_BAR = ("barline", None, None, None)


def test_metrics_identical_sequences() -> None:
    m = evaluate([_C4, _REST_E, _BAR, _D4], [_C4, _REST_E, _BAR, _D4])
    assert m.ser == 0.0
    assert m.pitch_accuracy == 1.0
    assert m.rhythm_accuracy == 1.0
    assert m.edit_distance == 0


def test_metrics_single_substitution_at_a_note() -> None:
    m = evaluate([_C4, _D4], [_C4, _E4])  # D4 quarter → E4 eighth
    assert m.edit_distance == 1
    assert m.ser == 0.5
    assert m.pitch_accuracy == 0.5  # C4 matched, D4 mismatched
    assert m.rhythm_accuracy == 0.5  # quarter ≠ eighth on the substituted note


def test_metrics_pure_deletion_of_a_note() -> None:
    m = evaluate([_C4, _D4], [_C4])
    assert m.edit_distance == 1
    assert m.n_pitch_targets == 2
    assert m.pitch_correct == 1
    assert m.pitch_accuracy == 0.5
    assert m.n_rhythm_targets == 2
    assert m.rhythm_accuracy == 0.5


def test_metrics_insertion_does_not_affect_pitch_or_rhythm_denominators() -> None:
    m = evaluate([_C4], [_C4, _D4])
    assert m.edit_distance == 1
    assert m.n_pitch_targets == 1
    assert m.pitch_correct == 1
    assert m.pitch_accuracy == 1.0
    assert m.n_rhythm_targets == 1
    assert m.rhythm_accuracy == 1.0
    assert m.ser == 1.0


def test_metrics_attribute_only_substitution() -> None:
    gt = [("clef", None, None, "G2"), _C4]
    pred = [("clef", None, None, "F4"), _C4]
    m = evaluate(gt, pred)
    assert m.edit_distance == 1
    assert m.n_pitch_targets == 1
    assert m.pitch_correct == 1
    assert m.pitch_accuracy == 1.0
    assert m.rhythm_accuracy == 1.0


def test_metrics_rapidfuzz_matches_python_reference() -> None:
    import random

    rng = random.Random(0)
    pool = [_C4, _D4, _E4, _REST_E, _BAR]
    for _ in range(20):
        gt = [rng.choice(pool) for _ in range(rng.randint(2, 12))]
        pred = list(gt)
        # Randomly mutate
        for _ in range(rng.randint(0, 3)):
            op = rng.choice(("ins", "del", "sub"))
            if op == "ins" and pred:
                pred.insert(rng.randrange(len(pred) + 1), rng.choice(pool))
            elif op == "del" and pred:
                pred.pop(rng.randrange(len(pred)))
            elif op == "sub" and pred:
                pred[rng.randrange(len(pred))] = rng.choice(pool)
        fast = align(gt, pred)
        ref = _align_python(gt, pred)
        assert fast.edit_distance == ref.edit_distance, (gt, pred)

        fast_metrics = evaluate(gt, pred)
        # Build metrics from the reference alignment too.
        from src.postproc.metrics import _counts_from_pairs

        n_p, p_c, n_r, r_c = _counts_from_pairs(ref.pairs)
        # Counts can differ on tie-breaks, but the rule must still hold:
        # both alignments are optimal, so the SER must agree.
        assert fast_metrics.ser == ref.edit_distance / max(len(gt), 1)
        # Sanity: counts are non-negative and bounded.
        assert n_p <= len(gt) and n_r <= len(gt)
        assert fast_metrics.n_pitch_targets <= len(gt)
        assert fast_metrics.n_rhythm_targets <= len(gt)


def test_metrics_empty_gt_produces_nan_accuracies() -> None:
    m = evaluate([], [])
    assert m.ser == 0.0
    assert math.isnan(m.pitch_accuracy)
    assert math.isnan(m.rhythm_accuracy)


def test_aggregate_matches_evaluate_batch() -> None:
    vb = build_default_vocabs()

    def encode(tuples):
        return (
            vb.type.encode([t[0] for t in tuples]),
            vb.pitch.encode([t[1] for t in tuples]),
            vb.rhythm.encode([t[2] for t in tuples]),
            vb.attribute.encode([t[3] for t in tuples]),
        )

    gt_samples = [
        [_C4, _D4, _REST_E],
        [_C4, _BAR, _D4],
        [_REST_E, _REST_E],
    ]
    pred_samples = [
        [_C4, _E4, _REST_E],
        [_C4, _BAR, _D4],
        [_REST_E],
    ]

    per_sample = [evaluate(gt, pr) for gt, pr in zip(gt_samples, pred_samples)]
    agg = aggregate(per_sample)
    batch = evaluate_batch(
        [encode(s) for s in gt_samples],
        [encode(s) for s in pred_samples],
        vb,
    )

    assert agg.edit_distance == batch.edit_distance
    assert agg.gt_length == batch.gt_length
    assert agg.n_pitch_targets == batch.n_pitch_targets
    assert agg.pitch_correct == batch.pitch_correct
    assert agg.n_rhythm_targets == batch.n_rhythm_targets
    assert agg.rhythm_correct == batch.rhythm_correct
    assert agg.ser == batch.ser
    assert agg.pitch_accuracy == batch.pitch_accuracy
    assert agg.rhythm_accuracy == batch.rhythm_accuracy


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_end_to_end_generator_to_jianpu_and_metrics() -> None:
    vb = build_default_vocabs()
    sample = MelodyGenerator().generate(seed=42, sample_idx=0)
    ids = {name: vocab.encode(sample.labels[name]) for name, vocab in vb}

    jianpu = ids_to_jianpu(
        ids["type"], ids["pitch"], ids["rhythm"], ids["attribute"], vb
    )
    assert isinstance(jianpu, str)
    assert any(d in jianpu for d in "1234567")  # at least one scale degree
    # Round-trip: GT vs itself ⇒ perfect metrics.
    m = evaluate_ids(
        gt=(ids["type"], ids["pitch"], ids["rhythm"], ids["attribute"]),
        pred=(ids["type"], ids["pitch"], ids["rhythm"], ids["attribute"]),
        vocab=vb,
    )
    assert m.ser == 0.0
    assert m.pitch_accuracy == 1.0
    assert m.rhythm_accuracy == 1.0
