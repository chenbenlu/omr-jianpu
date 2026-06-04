# Augmentation ablation — ViT photo fine-tune on real photographed scores

All rows: warm-start from `vit-528` (`checkpoints/vit-20260528-090804`), 3 epochs,
LR 5e-5, batch 48. Per condition the checkpoint is **selected on the external `val`
split (100 photos)** and reported on the held-out external **`test` split (100
photos)**. Clean-val = first 500 of `data/synthetic/val` (no photo augmentation).

| condition | aug profile | sel step | TEST SER | TEST pitch | TEST rhythm | clean SER | clean pitch |
|---|---|---|---|---|---|---|---|
| baseline (no fine-tune) | — | — | 0.634 | 0.325 | 0.862 | 0.0030 | 0.9983 |
| fine-tune, no photo aug | `default` | 4000 | 0.634 | 0.314 | 0.860 | 0.0016 | 0.9990 |
| **photo FULL** | `photo` | 6000 | **0.258** | **0.645** | 0.965 | 0.0083 | 0.9917 |
| photo − geometric | `photo_no_geom` | 2000 | 0.605 | 0.331 | 0.817 | 0.0034 | 0.9980 |
| photo − lighting | `photo_no_light` | 2000 | 0.326 | 0.563 | 0.948 | 0.0117 | 0.9859 |
| photo − degrade | `photo_no_degrade` | 6000 | 0.273 | 0.634 | 0.959 | 0.0087 | 0.9910 |

## Leave-one-out contribution (FULL minus the run with that group removed)

| group removed | Δ SER (worse when removed) | Δ pitch (lost when removed) |
|---|---|---|
| geometric (rotate/perspective) | +0.347 | −0.314 |
| lighting (brightness/gamma/shadow) | +0.068 | −0.082 |
| degrade (blur/noise/JPEG) | +0.015 | −0.011 |

## Conclusions

1. **Geometric warps (rotation/perspective) are the dominant lever.** Removing them
   collapses photo performance almost back to baseline (SER 0.605, pitch 0.331 vs
   baseline 0.634 / 0.325). This empirically confirms the error diagnosis: the
   sim→real gap is mostly camera-angle skew causing vertical mis-registration
   (the 78% "same octave, wrong step" pitch errors).
2. **The gain is from augmentation, not extra training.** The `default` (no photo
   aug) fine-tune matches baseline on photos (SER 0.634, pitch 0.314) — and even
   sharpens clean val slightly. So more synthetic steps alone do nothing for photos.
3. **Ranking: geometric ≫ lighting > degrade.** Lighting adds a useful ~0.07 SER /
   ~0.08 pitch; blur/noise/JPEG contribute little (~0.015 SER) — the phone captures
   are fairly sharp, and heavy blur risks erasing the thin staff lines pitch needs.
4. **Clean-val forgetting is small and tracks adaptation.** The more a run adapts to
   photos (geometric+lighting), the more it forgets clean (FULL: pitch 0.9983→0.9917,
   −0.7 pt; SER 0.0030→0.0083). The − geometric run barely adapts, so it barely
   forgets. The trade is heavily favourable: +32 pt photo pitch for −0.7 pt clean pitch.

## Implications
- Keep geometric central; consider a slightly stronger geometric variant (more
  perspective/rotation diversity) and trim the degrade group.
- Geometric's importance (a vertical-registration problem) is also the strongest
  argument for trying the CRNN+CTC encoder (H=128, full width) which preserves the
  vertical resolution ViT's 224² resize discards.
- To curb the minor clean-val forgetting, mix a fraction of `default`-aug (clean)
  samples into the fine-tune, or lower LR.
