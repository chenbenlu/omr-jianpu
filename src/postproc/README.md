# `src.postproc` — Post-processing (Owner: Member C)

Translates the decoder's **four decoupled output streams** (`type` / `pitch` /
`rhythm` / `attribute`) into rendered Jianpu, and provides the DP-aligned
evaluation metrics Member B uses to score VED architectures during training.
Pure Python; no GPU dependency.

## Module map

| File | Purpose |
|------|---------|
| `decode.py` | ID streams → list of `(type, pitch, rhythm, attribute)` tuples. Type-anchored, tolerant of per-head disagreement at prediction time. |
| `jianpu.py` | Token-tuple stream → ASCII Jianpu text. State machine over running clef / key signature. |
| `metrics.py` | DP Levenshtein alignment on 4-tuples + decoupled accuracies. Hot path uses `rapidfuzz` (C++). |

## Public API

```python
from src.postproc import (
    ids_to_tuples,         # decode 4 ID streams → list of TokenTuple
    ids_to_jianpu,         # one-shot: 4 ID streams → Jianpu text
    tuples_to_jianpu,      # token tuples → Jianpu text
    JianpuRenderConfig,
    evaluate_ids,          # per-sample metrics from ID streams
    evaluate_batch,        # validation-loop entry point: list[IdSeqs] → EvalMetrics
    evaluate, align, aggregate,
    AlignmentResult, EvalMetrics, TokenTuple,
)
```

`TokenTuple` is `(type, pitch, rhythm, attribute)`. `pitch` / `rhythm` /
`attribute` are `None` at `<NULL>` positions and `"<UNK>"` when the model
emits an out-of-vocabulary or wrong-special token (surfaced, not dropped).

## Decode: type-anchored, prediction-tolerant

Ground-truth streams from `src.data` are lock-step across all four heads.
**Predictions are not** — each decoder head emits independently, so early
in training the model regularly produces e.g. `type=note` at a position
where `pitch=<PAD>` or `attribute=<EOS>`. `ids_to_tuples` uses the `type`
stream as the structural authority:

1. **EOS truncation** at the first `type_ids[i] == EOS_ID` (cuts all four).
2. **Leading BOS** stripped if `type_ids[0] == BOS_ID`.
3. **Position-by-position**:
   - `type_ids[i] ∈ {PAD, BOS, EOS}` mid-sequence → drop the whole position.
   - `type_ids[i] == UNK` → emit type as `"<UNK>"`.
   - For each of `pitch / rhythm / attribute`: PAD/BOS/EOS at a kept
     position → coerce to `"<UNK>"`; NULL → `None`; UNK → `"<UNK>"`.

Pass `strict=True` to raise `ValueError` on any cross-stream special-token
disagreement — useful as a debug assertion against GT streams.

## Jianpu text format

ASCII, monospace-friendly, grep-friendly. Default `JianpuRenderConfig`:

- **Octave marks** — apostrophe up, comma down, repeated per octave offset:
  `1` (central), `1'` (one up), `1,,` (two down).
- **Rhythm** — underline prefix + dash suffix:

  | Token   | Output |
  |---------|--------|
  | `whole`        | `1 - - -` |
  | `half`         | `1 -` |
  | `quarter`      | `1` |
  | `eighth`       | `_1` |
  | `16th`         | `__1` |
  | `32nd`         | `___1` |
  | `*_dot`        | append ` .` after the number/dashes |

- **Accidentals** — `#`, `##`, `b`, `bb` immediately before the degree.
- **Rest** — `0` with the same underline/dash treatment as a note.
- **Barline** — ` | `.
- **Headers** (when `emit_header=True`) — first clef/key/time emits a
  bracketed prefix line: `[Clef: G2] [Key: G major] [Time: 4/4]`.

The renderer tracks running state: `clef` updates the central octave
(treble=4, bass=3), `key_signature` updates the tonic (fifths in `-7..+7`),
`time_signature` is shown in the header but doesn't change body rendering.

### Context-aware UNK rendering

| Field with UNK              | Rendering                                       |
|-----------------------------|-------------------------------------------------|
| `type`                      | standalone `?`                                  |
| `pitch` (type=`note`)       | `{underline}?{rhythm_tail}` — keeps rhythm shape |
| `rhythm` (type=`note/rest`) | `{degree}?` — bare body + marker, no underline/dashes |
| `attribute` (clef/key/time) | state unchanged; header shows `[Clef: ?]` etc.  |

## Metrics — DP-aligned decoupled accuracy

Image-to-sequence models make insertion/deletion errors, so element-wise
`pred[i] == gt[i]` would catastrophically fail on a single missed token.
`metrics.py` runs a Levenshtein alignment over 4-tuples first
(substitution cost = 1 iff *any* field differs, which is exactly tuple
inequality), then computes:

- **SER** — `edit_distance / max(len(gt), 1)`.
- **Pitch accuracy** — denominator: positions where `gt.type == "note"`,
  including those aligned to a deletion. Numerator: same condition AND
  `pred is not None and pred.pitch == gt.pitch`.
- **Rhythm accuracy** — denominator: positions where
  `gt.type in {"note", "rest"}`. Same numerator rule with `pred.rhythm`.
- **Insertions** (pred-only positions) contribute to edit distance but
  not to pitch/rhythm denominators.
- Empty denominators yield `float("nan")` — `aggregate` and
  `evaluate_batch` sum counts before dividing so aggregation is correct
  even when individual samples have no notes/rests.

### Hot path

`evaluate_ids` runs once per validation sample (~8,700 samples / epoch).
The alignment uses
[`rapidfuzz.distance.Levenshtein.editops`](https://github.com/rapidfuzz/RapidFuzz)
operating directly on lists of 4-tuples — a full pass over 8,700 samples
finishes in ~0.18 s on one CPU core. A pure-Python reference DP
(`_align_python`) is retained as a test oracle.

`requirements.txt` pins `rapidfuzz>=3.6`.

## Usage

```python
from src.data import MelodyGenerator, build_default_vocabs
from src.postproc import ids_to_jianpu, evaluate_batch

vb  = build_default_vocabs()
gen = MelodyGenerator()

def encode(labels):
    return tuple(vocab.encode(labels[name]) for name, vocab in vb)

# Render a generated sample.
labels = gen.generate(seed=42, sample_idx=0).labels
type_ids, pitch_ids, rhythm_ids, attribute_ids = encode(labels)
print(ids_to_jianpu(type_ids, pitch_ids, rhythm_ids, attribute_ids, vb))

# Validation-loop entry point — counts aggregate correctly across samples.
gt_batch   = [encode(gen.generate(seed=1_000_000, sample_idx=i).labels)
              for i in range(N)]
pred_batch = run_decoder(...)   # Member B's beam/greedy decode → same shape

m = evaluate_batch(gt_batch=gt_batch, pred_batch=pred_batch, vocab=vb)
print(m.ser, m.pitch_accuracy, m.rhythm_accuracy)
```

## Tests

```bash
pytest tests/test_postproc.py -v
```

30 tests covering: GT decode round-trip, prediction-path coercion
(PAD/BOS/EOS in non-type streams, EOS truncation, mid-sequence PAD drop),
Jianpu correctness across C / G / bass-clef / key-change / UNK cases,
metrics edge cases (substitution / deletion / insertion / attribute-only),
rapidfuzz vs pure-Python cross-check, and `aggregate` ≡ `evaluate_batch`.

All CPU-only; no renderer / verovio / cairosvg invocation.
