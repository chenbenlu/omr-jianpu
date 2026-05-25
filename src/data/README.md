# `src.data` — Synthetic OMR Data Pipeline (Owner: Member A)

100% synthetic printed monophonic score generator. music21 builds a Stream →
verovio renders it to SVG → cairosvg rasterizes to PNG. Labels are emitted
directly from the generator (no SVG parse-back) as **four parallel decoupled
streams**: `type`, `pitch`, `rhythm`, `attribute`. Decoupling lets the model
team score pitch accuracy and rhythm accuracy independently when comparing
Vision-Encoder-Decoder architectures.

## File layout

| File | Responsibility |
|------|----------------|
| `vocabulary.py` | Four parallel `Vocabulary` instances bundled in `VocabBundle`; closed-set, built once via `build_default_vocabs()` |
| `generator.py` | `MelodyGenerator` — deterministic random monophonic music21 Stream + parallel labels keyed on `(seed, sample_idx)` |
| `renderer.py` | `StaffRenderer` — Stream → MusicXML → verovio SVG → PNG; lazy verovio init per worker |
| `encoders.py` | `EncoderSpec` dataclass + `ENCODER_REGISTRY` (`vit`, `resnet`) + Albumentations transform factories |
| `dataset.py` | `SyntheticOMRDataset` (on-the-fly) and `PreRenderedOMRDataset` (from disk) |
| `prerender.py` | CLI: materialize val/test split into `data/synthetic/{val,test}/` |
| `dataloader.py` | `create_dataloaders(encoder=..., ...)` factory + `collate_fn` (handles dynamic-width batches) |

## Public API

```python
from src.data import (
    Vocabulary, VocabBundle, build_default_vocabs, save_bundle, load_bundle,
    GeneratorConfig, MelodyGenerator, GeneratedSample,
    RenderConfig, StaffRenderer,
    EncoderSpec, ENCODER_REGISTRY, get_encoder_spec,
    SyntheticOMRDataset, PreRenderedOMRDataset,
    create_dataloaders, collate_fn,
    prerender_split,
)
```

## Batch contract (frozen)

```python
{
  "pixel_values":           torch.Tensor,      # (B, C, H, W) — per encoder spec
  "type_ids":               torch.LongTensor,  # (B, L)
  "pitch_ids":              torch.LongTensor,  # (B, L) — NULL_ID where type != note
  "rhythm_ids":             torch.LongTensor,  # (B, L) — NULL_ID where type ∈ {barline, clef, key/time sig}
  "attribute_ids":          torch.LongTensor,  # (B, L) — NULL_ID where type ∈ {note, rest, barline}
  "decoder_attention_mask": torch.LongTensor,  # (B, L) — 1=real, 0=PAD
  "label_lengths":          torch.LongTensor,  # (B,)   — same across all four label streams
}
```

The four label tensors are always **aligned** — one row per emitted symbol.
Specials are shared across vocabs: `PAD=0, BOS=1, EOS=2, UNK=3, NULL=4` (the
`type` vocab has no NULL since every symbol has a type).

## Encoder transforms

Picked by name or by passing a custom `EncoderSpec`:

```python
from src.data import create_dataloaders, EncoderSpec

# Built-ins:
create_dataloaders(out_dir, encoder="vit", ...)     # (B, 3, 224, 224)  fixed
create_dataloaders(out_dir, encoder="resnet", ...)  # (B, 1, 128, W)    dynamic

# Custom:
spec = EncoderSpec(name="custom", channels=1, target_height=64,
                   target_width=512, max_width=512,
                   normalize_mean=(0.5,), normalize_std=(0.5,))
create_dataloaders(out_dir, encoder=spec, ...)
```

Dynamic-width samples are aspect-preserving (height-locked), each emitting
`(C, H, W_i)`. `collate_fn` pads along W to the batch-max, filling with `+1.0`
(normalized white). Samples larger than `max_width` are center-cropped at the
transform step to protect against OOM blowups.

## Sample storage

- **Train** is on-the-fly: each `__getitem__(i)` calls `generator.generate(seed, i)`
  → renderer → encode. The `(seed, sample_idx)` pair fully determines the
  sample, so the same `seed` reproduces the same data deterministically.
- **Val / test** are pre-rendered once with the `prerender` CLI for reproducible
  evaluation; the manifest lives at `data/synthetic/{val,test}/manifest.jsonl`.
  If no pre-rendered dir is supplied, `create_dataloaders` falls back to
  on-the-fly with a disjoint seed offset (handy for unit tests).

```bash
python -m src.data.prerender --out data/synthetic/val  --n 1000 --seed 1000000 --split val
python -m src.data.prerender --out data/synthetic/test --n 1000 --seed 2000000 --split test
```

The vocab bundle is closed-set and persisted to
`<out_dir>/vocab/{type,pitch,rhythm,attribute}.json` on the first
`create_dataloaders` call.

## Symbol scope

Monophonic only — **no chords**, **no slurs**, **no handwriting variations**.
The renderer asserts on any `chord.Chord` in the stream as a defensive check.

- **type**: `note`, `rest`, `barline`, `clef`, `key_signature`, `time_signature`
- **pitch** (notes only): `C–B × {bb, b, _, #, ##} × {2..6}` + `<NULL>`
- **rhythm** (notes/rests only): `whole/half/quarter/eighth/16th/32nd` (× `_dot`) + `<NULL>`
- **attribute** (clef/key/time only): clef name (`G2/F4/C3/C4`), key sig fifths
  (`ks-7..ks+7`), time sig (`2/4..12/8`) + `<NULL>`

## Commands

```bash
# Tests (CPU only; renderer is mocked so verovio/cairosvg never invoked in CI):
pytest tests/test_data.py -v

# Lint and format:
ruff check src/data/ tests/test_data.py
black --check src/data/ tests/test_data.py

# Smoke an end-to-end batch (requires libcairo2 + verovio installed):
python -c "
from src.data import create_dataloaders
loaders = create_dataloaders(out_dir='data/synthetic', encoder='vit',
                             train_size=4, batch_size=2, num_workers=0)
b = next(iter(loaders['train']))
print(b['pixel_values'].shape)  # torch.Size([2, 3, 224, 224])
print(b['type_ids'].shape)
"
```

## Dependencies

- `music21>=9.1` — Stream construction + MusicXML serialization
- `verovio==6.2.1` (pinned exactly — minor versions change glyph spacing)
- `cairosvg>=2.7,<3.0` (needs `libcairo2` system library)
- `albumentations>=1.4`, `opencv-python-headless`, `pillow`, `numpy`, `torch`

## Design notes

- **Determinism**: `MelodyGenerator.generate` derives its RNG from
  `np.random.SeedSequence([seed, sample_idx])` — collision-free hierarchical
  seeding. Same `(seed, idx)` always yields the same labels and stream.
- **Monophony is structural**: the generator only calls
  `note.Note(single_pitch)` or `note.Rest`, never `chord.Chord`, never `Voice`,
  never `insert` with explicit offsets. The renderer asserts a chord-free
  stream as a defensive check.
- **Verovio worker safety**: `StaffRenderer._toolkit` is `None` until first
  use and is dropped on `__getstate__` so DataLoader workers (fork or spawn)
  always rebuild a clean per-process toolkit. The C++ SWIG-wrapped toolkit is
  not picklable.
- **Labels are ground truth**: the generator emits parallel label lists in
  lock-step with stream `append`s. The renderer is one-way — we never parse
  SVG back into labels.
- **Closed-vocab UNK is a bug**: the dataset asserts `UNK_ID` does not appear
  in any encoded stream. If it does, the generator emitted a token outside the
  default vocab (extend `vocabulary.py`).
