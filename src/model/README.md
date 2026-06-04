# `src.model` — Model Training (Owner: Member B)

Vision-Encoder-Decoder definition, loss functions, optimizer / scheduler setup,
and the training loop live here. Consumes batches from `src.data` (4-stream
decoupled labels: `type` / `pitch` / `rhythm` / `attribute`) and emits
checkpoints (gitignored) plus per-stream logits.

## Encoder choices

The data layer serves either `vit` (3-channel 224×224 fixed) or `resnet`
(1-channel grayscale, H=128, dynamic W) batches; pick via the `encoder` kwarg
of `src.data.create_dataloaders`. The decoder is shared: a Transformer with
**four output heads** (one per label stream) so we can measure pitch accuracy
and rhythm accuracy independently when comparing encoders.

## Encoder + decoder ablation

Same four-stream decoupled labels and same val split; the variable is the
encoder–decoder pairing. Full 100k-sample × 30-epoch runs on the 1,000-sample
held-out synthetic split:

| Encoder | Decoder / loss | val SER ↓ | pitch acc ↑ | rhythm acc ↑ |
|---|---|---|---|---|
| `vit` (pretrained, `[CLS]` dropped) | AR multi-head BART | **0.0029** | **99.85%** | 99.78% |
| `resnet` (from-scratch CNN) | AR multi-head BART | ~1.10 | **~0%** | ~30–40% |
| `resnet` (same encoder) | **CRNN + per-head CTC** (BiLSTM + 4 heads) | **0.0000** | **100%** | **100%** |

ViT+AR solves the task. ResNet+AR fails on pitch: its per-position pitch
cross-entropy is pinned at ≈1.25 throughout training — exactly the floor of
a head that predicts `<NULL>` correctly on non-note positions and is
uninformative on notes (≈ `P(note) · ln |pitch vocab|`), i.e. zero pitch
signal. The third row, swapping only the decoder for a BiLSTM + per-head
CTC, solves the task perfectly on the *same* column-pooled CNN features
that the AR decoder cannot read for pitch.

**Diagnosis — the failure is the AR-over-1-D-features pairing, not the
encoder alone.** The original explanation (translation-equivariant CNN
collapses image height into one token per column, losing absolute vertical
position which is what pitch is) is the right description of what the
ResNet encoder discards, but it is *not* the binding constraint. CTC's
flexible alignment removes the one-column-one-symbol assumption that AR
cross-attention enforces, and the BiLSTM's bidirectional context lets each
column draw on neighbouring columns' staff-line geometry as a vertical
reference frame. Once both are in place, pitch recovers fully — invariant
to learning rate, vertical resolution, or explicit vertical pos-emb (none
of those interventions moved the AR pitch floor either).

**Status on `main`.** Only the AR architectures live in this branch:
`vit` is the production config; `resnet` is retained as the documented
negative example of the broken pairing. The CRNN+CTC variant lives on the
`feature/B-crnn-and-ctr` branch (`configs/model/crnn.yaml`, `use_ctc=true`)
— empirically validated end-to-end at 100k samples but not yet integrated
with `src/deploy/`, so it has not been merged.

## Sample-efficiency sweep: how much data does each architecture need?

Same decoder/loss/optimizer/30-epoch budget, sweeping `data.train_size` at
five log-spaced points. CRNN+CTC numbers are from the `feature/B-crnn-and-ctr`
branch (`model=crnn`); ViT-AR numbers are from this branch (`model=vit`).
Best val SER across 30 epochs on the same 1,000-sample held-out split:

| train_size | CRNN+CTC SER | CRNN pitch | ViT-AR SER | ViT pitch |
|-----------:|-------------:|-----------:|-----------:|----------:|
| 1,000   | 0.948  | 0.0%   | 1.07   | 1.1%   |
| 5,000   | **0.067**  | 97.1%  | 0.866  | 3.7%   |
| 20,000  | **0.010**  | 99.95% | 0.894  | 0.2%   |
| 50,000  | **0.0002** | 100%   | **0.032**  | 97.5%  |
| 100,000 | **0.000**  | 100%   | **0.0029** | 99.85% |

CRNN+CTC reaches usable transcription quality already at 5k samples;
ViT-AR's pitch head fails to learn at all until 50k samples and shows a
sharp data threshold between 20k and 50k. **This inverts the common
expectation that ImageNet pretraining helps in the low-data regime** — the
pretrained ViT prior carries no music-notation structure, and the ~310-way
pitch vocabulary needs roughly 1k–2k examples per class before the AR
pitch head escapes the NULL-only loss floor. CTC's flexible alignment +
BiLSTM context make per-symbol learning proportionately more
sample-efficient.

Outputs in `reports/scaling/`: `summary.csv`, `summary.md`,
`val_ser_vs_train_size.png`, `val_pitch_acc_vs_train_size.png`, and a
paper-ready paragraph in `paper_paragraph.md`.

## Training data path

Train defaults to **on-the-fly** rendering (`SyntheticOMRDataset`): verovio
re-renders every image each epoch, which is CPU-bound and leaves the GPU ~90%
idle. For GPU-bound training, pre-render the split once and read PNGs from disk
via the `data.train_dir` config key:

```bash
python -m scripts.prerender_train --out data/synthetic/train \
    --n 100000 --seed 42 --workers 8        # ~8 min, sharded across workers
python -m src.model.train data.train_dir=data/synthetic/train \
    data.batch_size=256                     # GPU util ~99%, ~7× faster
```

Augmentation is still applied at load time, so the training distribution is
unchanged; only the (deterministic, `(seed, idx)`-keyed) base render is cached.

## Reference training configs

Commands that reproduce the ablation numbers above on a single RTX 5070
(12 GB), on-the-fly rendering.

### ViT — val SER 0.0029

```bash
python -m src.model.train \
    model=vit \
    data.train_size=100000 data.batch_size=32 data.num_workers=4 \
    train.epochs=30 train.eval_every_steps=3125 train.log_every_steps=100 \
    train.gen_max_length=64 train.mixed_precision=bf16 \
    optim.optimizer.lr=5e-4 optim.scheduler.num_warmup_steps=2000 \
    logging=tensorboard
```

~7 h wall-clock, VRAM ~4.5 GB. SER 0.90 → 0.07 by epoch 5 → **0.0029 best at
epoch 22**, then plateaus through epoch 30 — use the epoch-22 checkpoint, not
the final epoch. Expect ~17 GB of `accelerate.save_state` checkpoints to
accumulate over the run; prune intermediate `step-*-best` dirs periodically.

### ResNet ablation

```bash
python -m src.model.train \
    model=resnet \
    data.train_size=100000 data.batch_size=128 data.num_workers=8 \
    train.epochs=30 train.eval_every_steps=782 train.log_every_steps=50 \
    train.gen_max_length=64 train.mixed_precision=bf16 \
    optim.optimizer.lr=1e-3 optim.scheduler.num_warmup_steps=500 \
    logging=tensorboard
```

~3 h wall-clock, VRAM ~6.2 GB.

## Batch contract

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

Shared special IDs across all four vocabs: `PAD=0, BOS=1, EOS=2, UNK=3, NULL=4`
(the `type` vocab has no NULL). Set each head's `pad_token_id` /
`decoder_start_token_id` / `eos_token_id` / `vocab_size` accordingly; do NOT
manually `shift_tokens_right` (HF does it internally).

Per-stream cross-entropy losses are summed (or weighted) into the training
objective; NULL positions can be masked out of the pitch / rhythm / attribute
losses depending on whether the head should learn to predict NULL or simply
ignore those positions.
