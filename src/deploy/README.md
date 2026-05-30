# `src.deploy` — Integration & Deployment (Owner: Member D)

End-to-end pipeline glue, inference API, Streamlit demo, and experiment
reporting. Imports from `src.data`, `src.model`, and `src.postproc` to wire the
full path **image → encoder transform → VED (4-head decoder) → postproc tuples
→ Jianpu / engraved staff**. No training-time dependencies.

## Inference flow

1. Take a user-uploaded staff image (or a pre-rendered val sample).
2. Apply the matching `EncoderSpec` eval transform from
   `src.data.get_encoder_spec(...)` to produce a `pixel_values` tensor.
3. Run `OMRModel.generate(...)`; receive four decoupled ID streams.
4. `src.postproc.ids_to_tuples` → structured `TokenTuple`s (the canonical
   intermediate everything else consumes).
5. From the tuples, present any of:
   - **Compact ASCII Jianpu** (`ids_to_jianpu`, frozen single-line format).
   - **Pretty Jianpu** — 2-D engraved-looking notation (HTML/CSS grid in the
     UI, SVG for files/CLI, 3-row monospace for terminals).
   - **Engraved staff PNG** — reconstruct a `music21.Stream` and render through
     the project's existing verovio path.

## Public API (`from src.deploy import …`)

```python
OMRInferencer          # load checkpoint once; predict()/predict_batch()
JianpuPrediction       # dataclass: jianpu(str), tuples, raw ids, length

jianpu_html(tuples)    # CSS-grid Jianpu, for st.markdown(unsafe_allow_html=True)
jianpu_svg(tuples)     # standalone SVG, for file write / CLI export
pretty_jianpu(tuples)  # 3-row monospace fallback (terminal-friendly)

tuples_to_stream(t)    # inverse of the generator → music21.Score
render_staff_png(t)    # (np.ndarray, backend) — lilypond if on PATH, else verovio
which_backend(b='auto')
lilypond_available()
```

## Inferencer construction

```python
from src.deploy import OMRInferencer
inf = OMRInferencer(
    "checkpoints/vit-20260528-090804",  # may be a run dir OR a step-N-best dir
    encoder=None,        # default: parse leading "vit"/"resnet" from the dir name
    device=None,         # default: cuda if available else cpu
    model_config=None,   # must match training; default = configs/model/vit.yaml shape
    max_length=64,
)
pred = inf.predict(image)         # image: PIL.Image | np.ndarray | path
```

Key behaviours that demo robustness depends on:

- **Encoder inference from dir name**: nothing is persisted inside the
  checkpoint, so the leading token of the run-dir name is authoritative. Pass
  `encoder=` explicitly to override.
- **Best snapshot auto-pick**: if `ckpt_dir` has no `model.safetensors` itself
  it picks the `step-N-best/` subdir with the highest `N`.
- **`predict` is total**: a wild prediction with an out-of-range key signature
  makes `ids_to_jianpu` raise; the inferencer catches it and falls back to an
  empty `pred.jianpu` rather than crashing the UI. `pred.tuples` is always
  valid (it goes through `ids_to_tuples(..., strict=False)`).

## Jianpu rendering layer (`jianpu_format.py`)

All three renderers share one structured per-symbol model:

```python
@dataclass(frozen=True)
class JianpuCell:
    body, accidental, dots_up, dots_down, underlines, tail
    is_barline, is_note, beat_group
```

Semantics (degree, accidental, octave, rhythm) are NOT re-derived — each
symbol is rendered through the frozen `src.postproc` single-symbol path and
the resulting compact token is decomposed, so the numbers always agree with
postproc's `ids_to_jianpu`.

### Beam grouping (continuous reduction lines within a beat)

`_cells()` accumulates each note's quarter-length and resets at barlines,
computing each note's `(measure, beat_index)` using `meter.TimeSignature.
beatDuration.quarterLength` for the running time signature. Notes in the same
(measure, beat) share a `beat_group`. `_beam_extends_right(cells, i, layer)`
joins beam layer L of cell i to cell i+1 iff:

- both are notes with `body != "0"` (rests carry no beams),
- both have `underlines >= layer`,
- they share `beat_group`.

This breaks beams cleanly across barlines, rests, beat boundaries, and layer
drops. Drawn in HTML by widening an `<i class="jp-x">` to bridge the
inter-column gap; in SVG by drawing the beam line from this cell's center to
`centers[i+1]`.

### Beam layer order

Layer 1 (8th, the longest/most-continuous line) sits **closest to the digit**;
extra beams (16th = layer 2, 32nd = layer 3) stack further below. HTML uses
CSS `grid-row: 1/2/3` to make this layout-independent (don't rely on flex
source order — `<i>` tags get inline-whitespace quirks in some sanitisers).

### Long-note alignment (`tail`)

A half/whole note's tail (` -`, ` - - -`, ` .`) sits **inside `jp-mid`** as an
absolutely-positioned `<span class="jp-tail" style="left:100%">`, so the tail
shares the digit's text baseline without inflating `jp-mid`'s box — octave
dots/beams (which are centred on the column) stay aligned to the **digit
only**. SVG mirrors this by drawing the digit at `text-anchor="middle" x=cx`
and the tail at `text-anchor="start" x=cx+9`, and growing per-cell width with
`len(tail)` so the next note doesn't overlap the extension.

## Engraving (`notation.py`)

`render_staff_png(tuples, backend="auto")` returns `(np.ndarray, backend_used)`.

- `tuples_to_stream` is the inverse of `src.data.generator`: walks the predicted
  tuples and rebuilds a `music21.Score` with explicit `Part` / `Measure` (split
  on `barline` tokens). **Don't return a flat Stream** — that forces music21's
  exporter to guess measure boundaries and the resulting MusicXML can be
  rejected by verovio in some host processes.
- Verovio path reuses [`StaffRenderer`](../data/renderer.py); a fresh toolkit is
  built per call (the SWIG C++ object can carry state from prior `loadData`).
  On failure, the Stream is normalised via `s.makeNotation()` and retried once.
- LilyPond path is dormant unless the `lilypond` binary is on PATH (no install
  in this repo). When present, `auto` picks it; otherwise verovio.
- `_ensure_verovio_resources()` calls `verovio.setDefaultResourcePath(<pkg>/data)`
  before rendering. Without this, verovio's auto-detection of the bundled
  Bravura/Leipzig fonts fails in some host processes (notably Streamlit),
  giving "`Bravura font could not be loaded`" → `loadData` returns False →
  the cryptic "verovio failed to parse the MusicXML payload" error.

## Streamlit demo

```bash
streamlit run src/deploy/app.py
```

Sidebar picks the checkpoint (any `checkpoints/<encoder>-...` dir), encoder
override (auto/vit/resnet), max decode length, and **Notation view**: Engraved
/ Jianpu (compact ASCII) / Jianpu (pretty). Upload a PNG or pick a val sample
(reads `data/synthetic/val/manifest.jsonl` to also show ground truth). The
inferencer is cached with `@st.cache_resource` keyed on `(ckpt, encoder)`.

## CLI

```bash
# inference (+ optional engraved staff + optional aligned-Jianpu SVG)
python -m scripts.predict_jianpu \
    --ckpt checkpoints/vit-20260528-090804 \
    --image data/synthetic/val/000000.png \
    --format compact \           # or pretty (3-row mono stdout)
    --svg out.svg \              # write aligned Jianpu SVG
    --engrave out.png            # write engraved staff PNG

# ablation report (auto-discovers encoders under runs/, emits figures + table)
python -m scripts.report_experiments --runs-dir runs --out reports/
```

## Tests

`tests/test_deploy.py` — all CPU, no real verovio/cairosvg invocation:
- inferencer construction, encoder-name resolution, best-snapshot pick
- `tuples_to_stream` structure + dotted/flat/holes
- engrave fallback (mocked `StaffRenderer.render`), `makeNotation` retry,
  font-path self-heal
- Jianpu cell decomposition, beam grouping (within beat / break at barline /
  break at rest), beam layer order, long-note digit alignment
- reporting script (discovers encoders, emits figures + summary)

```bash
pytest tests/test_deploy.py -v
ruff check src/deploy/ scripts/ tests/test_deploy.py
black --check src/deploy/ scripts/ tests/test_deploy.py
```
