# `src.data` — Camera-PrIMuS Data Pipeline (Owner: Member A)

Parses the Camera-PrIMuS staff-notation dataset, applies image augmentation, and
produces PyTorch `DataLoader` batches in the schema consumed by `src.model`. The
batch schema and special-token IDs below are a frozen cross-module contract —
changes require a coordinated PR with Member B.

## File layout

| File | Responsibility |
|------|----------------|
| `vocabulary.py` | `Vocabulary` class — token ↔ ID, build, save/load, `load_or_build` |
| `augmentation.py` | `get_train_augmentation()` / `get_eval_augmentation()` Albumentations pipelines |
| `splits.py` | Sample discovery, `Corpus/` root resolution, seeded train/val/test split |
| `dataset.py` | `PrIMuSDataset(torch.utils.data.Dataset)` and image-path resolution |
| `dataloader.py` | `create_dataloaders()` factory + `collate_fn` (batch contract lives here) |
| `download.py` | Standalone CLI to fetch and extract Camera-PrIMuS (~2 GB) |

## Public API

```python
from src.data import PrIMuSDataset, Vocabulary, create_dataloaders, collate_fn
```

- `create_dataloaders(data_dir, ...)` — returns `{"train", "val", "test"}` DataLoaders.
- `PrIMuSDataset(root, vocab, sample_ids, transform=, use_camera=, max_seq_len=)` — single-split dataset.
- `Vocabulary` — `build`, `load`, `save`, `load_or_build`, `encode`, `decode(ids, skip_special_tokens=False)`. Exposes `PAD_ID=0`, `BOS_ID=1`, `EOS_ID=2`, `UNK_ID=3`. Pass `skip_special_tokens=True` at inference time to drop PAD/BOS/EOS (UNK is preserved as a real prediction).
- `collate_fn(batch)` — pads to batch-max and produces the schema below.

## Batch schema (frozen contract)

```python
{
  "pixel_values":           torch.Tensor,  # (B, 3, 128, 1024)  float32, TrOCR-normalized
  "labels":                 torch.Tensor,  # (B, L)              int64,  tokens + EOS + PAD (no BOS)
  "decoder_attention_mask": torch.Tensor,  # (B, L)              int64,  1 = real token, 0 = PAD
  "label_lengths":          torch.Tensor,  # (B,)                int64,  true length incl. EOS (no BOS)
}
# Image normalization: mean = std = (0.5, 0.5, 0.5)  [TrOCR convention]
```

The mask is decoder-side — the ViT encoder consumes `pixel_values` directly
and needs no mask. Pass it through HuggingFace as the `decoder_attention_mask`
kwarg. `model(**batch)` does **not** work because `label_lengths` is not an
HF kwarg; pop it (or pass keys individually) before forwarding the batch.

### HF VisionEncoderDecoderModel wiring (Member B)

Five config keys must be set before the first `forward`, otherwise loss masking
and generation break in subtle ways:

```python
from src.data import Vocabulary

model.config.pad_token_id           = Vocabulary.PAD_ID   # 0 — masks PAD positions in CE loss
model.config.decoder_start_token_id = Vocabulary.BOS_ID   # 1 — HF prepends this via shift_tokens_right
model.config.eos_token_id           = Vocabulary.EOS_ID   # 2 — stops generate() early
model.config.vocab_size             = len(vocab)          # includes the four specials
model.decoder.resize_token_embeddings(len(vocab))         # only if loading a pretrained decoder
```

Do **not** pass `labels` through a manual `shift_tokens_right`. HF does the
shift internally; double-shifting yields `[BOS, BOS, tok1, …]`.

## Dataset format

Camera-PrIMuS unpacks to `data/raw/primus/Corpus/`:

```
data/raw/primus/
└── Corpus/
    └── {sample_id}/
        ├── {sample_id}.png              clean rendered staff image
        ├── {sample_id}_distorted.jpg    camera-distorted variant
        └── {sample_id}.semantic         tab-separated token sequence
```

Example `.semantic` line:

```
clef-G2	keySignature-EbM	timeSignature-3/4	note-Bb5_quarter	barline	...
```

## Commands

Download the dataset (≈ 2 GB, idempotent — skips if already extracted):

```bash
python -m src.data.download --dest data/raw/primus
```

Run the data tests (CPU only, no real dataset required):

```bash
pytest tests/test_data.py -v
```

Lint and format:

```bash
ruff check src/data/ tests/test_data.py
black --check src/data/ tests/test_data.py
```

Integration smoke (after download):

```bash
python -c "
from src.data import create_dataloaders
loaders = create_dataloaders('data/raw/primus', batch_size=4, num_workers=0)
batch = next(iter(loaders['train']))
print(batch['pixel_values'].shape)  # torch.Size([4, 3, 128, 1024])
print(batch['labels'].shape)        # torch.Size([4, L])
print(batch['label_lengths'])
"
```

### Real-dataset numbers (Camera-PrIMuS, 2026-05-24, `batch_size=8, num_workers=0`)

| metric | value |
|---|---|
| total samples | 87,678 |
| filtered for token-count > `max_seq_len - 1` (= 511) | 0 |
| split sizes (train / val / test) | 70,142 / 8,767 / 8,769 |
| vocab size (incl. 4 specials) | 1,716 |
| `create_dataloaders` cold (vocab build + save) | 2.0 s |
| `create_dataloaders` warm (vocab load from cache) | 1.5 s |
| first train batch (cold dataset cache) | 0.03 s |
| `pixel_values` range observed in batch | `[-0.88, +1.00]` |
| `label_lengths` in 8-sample batch | min 17, max 35, mean 25 |

Notes for Member B: the warm path's 1.5 s is dominated by `filter_overlong`'s
pass over 87 k `.semantic` files. With the current `max_seq_len=512` no sample
is ever filtered, but the read still happens every time — fine for one-off
DataLoader construction at the start of training, worth caching if you call
`create_dataloaders()` repeatedly.

## Design notes

- Images are resized with aspect-ratio preservation (`LongestMaxSize`) and
  white-paper padded (`fill=255`, top-left anchor) to a fixed `(3, 128, 1024)`.
  After TrOCR normalization the padded region maps to `+1.0`, matching the
  unprinted paper background so the encoder does not have to model "padding
  strip" as a distinct visual feature.
- Splits default to 80 % train / 10 % val / 10 % test by sample-ID seeded shuffle
  (`seed=42`). `splits.py` owns both the shuffle and the `Corpus/` subdirectory
  detection so `create_dataloaders` stays glue-only.
- Samples whose `.semantic` token count exceeds `max_seq_len - 1` are filtered
  at `create_dataloaders` time (one slot is reserved for EOS). Dropped IDs are
  logged at `WARNING` level. The truncation branch inside
  `PrIMuSDataset.__getitem__` is defensive — it only fires for callers that
  construct `PrIMuSDataset` directly and bypass the factory.
- The vocabulary is built from the **training split only**; tokens that appear
  only in val/test fall through to `<UNK>`. The built vocab is cached at
  `<corpus_root>/vocab.json` (alongside the sample directories) and re-used on
  subsequent runs — delete that file to force a rebuild.
- `use_camera=True` prefers `{sid}_distorted.jpg`, then `{sid}.jpg`, then falls
  back to `{sid}.png`. The order is encoded in `_resolve_image_path` so the test
  fixtures (which use plain `.jpg`) keep working alongside the real dataset.
- `collate_fn` pads each batch to its own max `label_length`, **not** to the
  global `max_seq_len`. The per-item `labels` tensor is still PAD-padded to
  `max_seq_len`; the collator trims it down.
