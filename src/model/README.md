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

## Encoder ablation: ViT works, ResNet has a structural pitch limitation

Same four-head decoder, same losses, same data — only the encoder differs
(full 100k-sample × 30-epoch runs, validated on the held-out synthetic set):

| Encoder | val SER ↓ | pitch acc ↑ | rhythm acc ↑ |
|---|---|---|---|
| `vit` (pretrained, `[CLS]` dropped) | **0.0029** | **99.85%** | **99.78%** |
| `resnet` (from-scratch CNN) | ~1.10 | **~0%** | ~30–40% |

ViT learns the task almost perfectly. The from-scratch ResNet encoder reaches
comparable *type* and partial *rhythm* accuracy but **never learns pitch** —
its per-position pitch cross-entropy is pinned at ≈1.25 throughout training.
That floor is exactly the loss of a head that predicts `<NULL>` correctly on
non-note positions and is uninformative on notes (≈ `P(note) · ln |pitch
vocab|`), i.e. zero pitch signal.

**Diagnosis — a representation limitation, not a tuning one.** `ResNetEncoder`
collapses the image *height* into a single token per image column
(`feat.mean(dim=2)` → a 1-D sequence over width). But a convolutional stack is
**translation-equivariant**: a notehead produces the same local features
regardless of *where* it sits vertically, so the column features encode "a note
is here" but not its **absolute vertical position** — and pitch *is* the
vertical position of the notehead on the staff. Collapsing height therefore
discards precisely the cue pitch depends on.

The ≈1.25 floor proved invariant to every intervention tried: learning rate
(5e-4 ↔ 1.4e-3), replacing the height mean-pool with a height *flatten*,
increasing vertical resolution (H÷32 → H÷8), and even adding an explicit
learned vertical positional embedding. The same decoder + loss reaches 99.85%
pitch with the ViT encoder, so the bottleneck is the ResNet encoder's 1-D
representation (and the from-scratch cross-attention alignment it forces), not
the decoder, the loss, or the data.

A principled fix would preserve 2-D structure — emit an `H_out × W_out` grid of
CNN-feature tokens with 2-D positional embeddings (a CNN-feature analogue of
ViT) rather than collapsing to columns. The current `ResNetEncoder` is retained
deliberately as the empirical demonstration of this limitation; **`vit` is the
working configuration.**

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
