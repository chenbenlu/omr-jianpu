"""ID → token-tuple decode, type-anchored across the four decoupled heads.

Ground-truth streams emitted by ``src.data`` are guaranteed position-aligned
across all four heads. Model predictions are not: each head emits
independently, so early in training the model regularly produces e.g.
``type=note`` at a position where ``pitch=<PAD>`` or ``attribute=<EOS>``.

The decode policy here uses the ``type`` stream as the structural authority:
the existence of a position in the output depends solely on the type token at
that index. Special-token leakage in the other three streams is coerced to
``<UNK>`` rather than silently dropping the position — which keeps the four
fields length-aligned and surfaces the model failure visibly downstream.
"""

from __future__ import annotations

from typing import Sequence

from src.data.vocabulary import VocabBundle, Vocabulary

TokenTuple = tuple[str, str | None, str | None, str | None]

_SPECIALS = frozenset({Vocabulary.PAD_ID, Vocabulary.BOS_ID, Vocabulary.EOS_ID})


def _decode_payload(tok_id: int, vocab: Vocabulary) -> str | None:
    """Decode a single id from a head that allows ``<NULL>``."""
    if tok_id in _SPECIALS:
        return "<UNK>"
    if vocab.has_null and tok_id == Vocabulary.NULL_ID:
        return None
    return vocab.id_to_token.get(int(tok_id), "<UNK>")


def ids_to_tuples(
    type_ids: Sequence[int],
    pitch_ids: Sequence[int],
    rhythm_ids: Sequence[int],
    attribute_ids: Sequence[int],
    vocab: VocabBundle,
    *,
    strict: bool = False,
) -> list[TokenTuple]:
    """Decode parallel ID streams into a list of ``(type, pitch, rhythm,
    attribute)`` tuples.

    The ``type`` stream is authoritative: positions where ``type_ids[i]`` is
    ``PAD``/``BOS``/``EOS`` (mid-sequence) are dropped. The first EOS in the
    type stream truncates all four streams. A leading ``BOS`` is stripped.

    ``strict=True`` raises ``ValueError`` whenever any non-type stream
    contains a special token at a kept position — used by tests on
    ground-truth streams to catch generator alignment bugs.
    """
    type_ids = list(type_ids)
    pitch_ids = list(pitch_ids)
    rhythm_ids = list(rhythm_ids)
    attribute_ids = list(attribute_ids)

    n = len(type_ids)
    if not (len(pitch_ids) == len(rhythm_ids) == len(attribute_ids) == n):
        raise ValueError(
            "ids_to_tuples: all four ID streams must have equal length, "
            f"got type={n} pitch={len(pitch_ids)} "
            f"rhythm={len(rhythm_ids)} attribute={len(attribute_ids)}"
        )

    # EOS truncation on the type stream.
    for i, tid in enumerate(type_ids):
        if tid == Vocabulary.EOS_ID:
            n = i
            break

    start = 1 if n > 0 and type_ids[0] == Vocabulary.BOS_ID else 0

    out: list[TokenTuple] = []
    type_vocab = vocab.type
    for i in range(start, n):
        t_id = type_ids[i]
        if t_id in _SPECIALS:
            if strict:
                raise ValueError(
                    f"strict decode: special token in type stream at idx {i}"
                )
            continue

        if t_id == Vocabulary.UNK_ID:
            t_tok: str = "<UNK>"
        else:
            t_tok = type_vocab.id_to_token.get(int(t_id), "<UNK>")

        if strict:
            for name, ids in (
                ("pitch", pitch_ids),
                ("rhythm", rhythm_ids),
                ("attribute", attribute_ids),
            ):
                if ids[i] in _SPECIALS:
                    raise ValueError(
                        f"strict decode: special token in {name} stream " f"at idx {i}"
                    )

        p_tok = _decode_payload(pitch_ids[i], vocab.pitch)
        r_tok = _decode_payload(rhythm_ids[i], vocab.rhythm)
        a_tok = _decode_payload(attribute_ids[i], vocab.attribute)
        out.append((t_tok, p_tok, r_tok, a_tok))

    return out
