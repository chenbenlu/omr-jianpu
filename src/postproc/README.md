# `src.postproc` — Post-processing (Owner: Member C)

Translates the decoder's **four decoupled output streams** (`type` / `pitch` /
`rhythm` / `attribute`) into rendered Jianpu via music-theory mapping rules.
Pure-Python; no GPU dependency. Consumed by `src.deploy` for the end-user UI.

## Input contract

The decoder emits four parallel ID sequences (per the `src.data` batch
schema). Decode each sequence with its matching `Vocabulary` from
`src.data.build_default_vocabs()` and zip them position-wise — every position
is a 4-tuple `(type, pitch, rhythm, attribute)` where `pitch` / `rhythm` /
`attribute` are `None` (= `<NULL>`) at non-applicable positions.

Special IDs to strip before mapping: `PAD=0, BOS=1, EOS=2` (use
`Vocabulary.decode(..., skip_special_tokens=True)`). `UNK=3` is a real model
prediction — surface it (e.g. render as `?`) rather than silently dropping.

## Mapping responsibilities

Given the inferred clef / key / time signature (from the `attribute` stream at
the score head), translate each `note` symbol's `pitch` token (e.g. `F#5`) and
`rhythm` token (e.g. `eighth_dot`) into the corresponding Jianpu glyph + rhythm
notation. `rest` / `barline` map directly. Clef / key / time changes mid-piece
update the running mapping state.
