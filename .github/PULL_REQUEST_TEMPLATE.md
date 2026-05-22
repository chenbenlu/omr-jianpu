<!--
Branch name must follow:  <type>/<owner-letter>-<kebab-slug>
e.g.  feature/B-train-vit-decoder-baseline   fix/A-augment-rotation-clipping
See docs/GIT_WORKFLOW.md for the full policy.
-->

## What changed
<!-- One or two sentences. -->

## Why / which task
<!-- Link to the CLAUDE.md task or proposal section this addresses. -->

## How it was tested
<!-- Exact command(s) run inside `make shell`, or a link to a notebook cell with the result. -->
- [ ] `ruff check .` passes
- [ ] `black --check .` passes
- [ ] `pytest` passes (or N/A — explain below)
- [ ] Manual verification (describe):

## Module ownership
<!-- Tick all that this PR touches. Cross-module changes require an approving review from each affected owner. -->
- [ ] A — `src/data` (Data)
- [ ] B — `src/model` (Training)
- [ ] C — `src/postproc` (Post-processing)
- [ ] D — `src/deploy` (Integration / Deploy)
- [ ] Repo-wide infra (Docker, CI, configs, docs)

## Reviewer checklist
- [ ] Branch was rebased onto latest `main` (not merged)
- [ ] No model weights, datasets, or `.env` files are included in the diff
- [ ] PR will be **squash-merged**
