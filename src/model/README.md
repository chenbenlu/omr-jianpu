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
