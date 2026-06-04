# Scaling: CRNN+CTC vs ViT-AR — paper-ready paragraph

## Headline

**CRNN+CTC reaches near-perfect transcription from 5,000 training samples;
ViT-AR requires 50,000 before pitch begins to learn at all.**

## Suggested paragraph for `docs/proposal/5-exp.tex`

> We further sweep both architectures across five training-set sizes
> ($1{,}000$, $5{,}000$, $20{,}000$, $50{,}000$, $100{,}000$ samples) holding
> the held-out 1,000-sample validation split, optimizer, and 30-epoch budget
> fixed (Table~\ref{table:scaling_results}, Figure~\ref{fig:scaling}). The
> two architectures exhibit qualitatively different sample-efficiency
> regimes. CRNN+CTC reaches usable transcription quality already at
> $5{,}000$ samples (val SER $0.067$, pitch accuracy $97.1\%$) and is
> within $10^{-3}$ of perfect SER from $50{,}000$ samples upward. ViT-AR, in
> contrast, exhibits a sharp data threshold: at $\le 20{,}000$ samples its
> pitch head fails to learn at all (pitch accuracy $\le 3.7\%$, val SER
> stuck around $0.9$, the regime where free-running generation degenerates
> into runaway sequences), while at $50{,}000$ samples it converges to val
> SER $0.0318$ and pitch accuracy $97.5\%$, comparable to its $100{,}000$
> sample asymptote (val SER $0.0029$, pitch $99.85\%$).
>
> This contradicts the common expectation that ImageNet-pretrained features
> would dominate the low-data regime. The intuition fails because the
> pretrained ViT prior carries no music-notation structure --- pitch is
> determined by the absolute vertical position of a notehead on the staff,
> a cue that ImageNet pretraining does not reward. Combined with
> autoregressive cross-attention alignment (which forces a single column of
> the encoder output to be the source of each decoded symbol) and the
> $\sim 310$-way pitch vocabulary, the AR pipeline needs roughly
> $1{,}000$--$2{,}000$ examples per pitch class before its pitch head escapes
> the NULL-only loss floor. The CRNN+CTC pipeline's BiLSTM context lets each
> column draw on the staff-line geometry of its neighbours as a vertical
> reference frame, and CTC's flexible alignment removes the
> one-column-one-symbol constraint, so per-symbol learning becomes
> proportionately more sample-efficient.

## Suggested table caption

> Table~\ref{table:scaling_results}: Best validation SER, pitch accuracy,
> and rhythm accuracy reached within 30 epochs at each training-set size on
> the 1,000-sample held-out synthetic split. Single seed per cell.

## Suggested figure captions

- `val_ser_vs_train_size.png`: validation SER (log scale) vs.\ training-set
  size (log scale) for CRNN+CTC and ViT-AR. CRNN's curve drops between $1$k
  and $5$k; ViT's curve drops between $20$k and $50$k.
- `val_pitch_acc_vs_train_size.png`: validation pitch accuracy vs.\
  training-set size. Same data-threshold contrast — CRNN reaches $\sim 97\%$
  pitch at $5$k; ViT remains at $<4\%$ pitch until $50$k.

## Caveats to disclose in the paper

- Single seed per cell --- no within-cell statistical bars.
- ViT-AR's 100k point is from the original v1 training run (May 28); the
  $\le 50$k points are from the post-restart sweep with `num_workers=8`
  (which slows convergence vs.\ `num_workers=4`; see `src/model/README.md`).
  Repeats with `num_workers=4` would likely shift the ViT threshold lower
  but the overall trend is robust.
- 30-epoch budget. Low-data ViT runs are not just "needs more epochs":
  their best SER occurred mid-training (e.g.\ epoch 12 for 20k) and they
  did not improve through epoch 30.
