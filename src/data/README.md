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
- `Vocabulary` — `build`, `load`, `save`, `load_or_build`, `encode`, `decode`. Exposes `PAD_ID=0`, `BOS_ID=1`, `EOS_ID=2`, `UNK_ID=3`.
- `collate_fn(batch)` — pads to batch-max and produces the schema below.

## Batch schema (frozen contract)

```python
{
  "pixel_values":   torch.Tensor,  # (B, 3, 128, 1024)  float32, TrOCR-normalized
  "labels":         torch.Tensor,  # (B, L)              int64,  BOS + tokens + EOS + PAD
  "attention_mask": torch.Tensor,  # (B, L)              int64,  1 = real token, 0 = PAD
  "label_lengths":  torch.Tensor,  # (B,)                int64,  true length incl. BOS+EOS
}
# Image normalization: mean = std = (0.5, 0.5, 0.5)  [TrOCR convention]
```

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

## Design notes

- Images are resized with aspect-ratio preservation (`LongestMaxSize`) and
  gray-fill padded (`fill=127`, top-left anchor) to a fixed `(3, 128, 1024)`.
- Splits default to 80 % train / 10 % val / 10 % test by sample-ID seeded shuffle
  (`seed=42`). `splits.py` owns both the shuffle and the `Corpus/` subdirectory
  detection so `create_dataloaders` stays glue-only.
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
