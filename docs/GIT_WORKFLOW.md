# Git Workflow — OMR-to-Jianpu

A lightweight GitHub Flow tuned for our 4-person split.

| Letter | Member | Module                | Path        |
| ------ | ------ | --------------------- | ----------- |
| A      | TBD    | Data Engineering      | `src/data`     |
| B      | TBD    | Model Training        | `src/model`    |
| C      | TBD    | Post-processing       | `src/postproc` |
| D      | TBD    | Integration & Deploy  | `src/deploy`   |

`main` is the only long-lived branch. Everything else is short-lived and merges back via PR.

## 1. Branch Naming

Format: `<type>/<owner-letter>-<kebab-slug>`

The owner letter makes module ownership visible at a glance in `git branch -a` and in the PR list.

| Type      | Purpose                                       | Examples                                                                                                 |
| --------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `feature` | New capability targeted for `main`            | `feature/A-data-primus-parser`, `feature/B-train-vit-decoder-baseline`, `feature/C-postproc-jianpu-mapper`, `feature/D-deploy-streamlit-ui` |
| `fix`     | Bug fix targeted for `main`                   | `fix/B-loss-nan-on-empty-batch`, `fix/A-augment-rotation-clipping`                                       |
| `exp`     | Throwaway experimentation; never merged       | `exp/B-cosine-lr-warmup-trial`, `exp/B-encoder-swap-resnet50`                                            |
| `docs`    | Documentation only                            | `docs/D-update-readme`                                                                                   |
| `chore`   | Tooling / deps / CI bumps                     | `chore/D-bump-pytorch-2.3`                                                                               |

Rules:
- Lowercase, kebab-case slug. No spaces, no underscores.
- Slugs read as a noun phrase, not a verb sentence (`fix/A-loader-shuffle-bug`, not `fix/A-fix-the-loader`).
- `exp/*` branches are personal scratch space — force-push is allowed and they are deleted after the winning idea is re-submitted as a `feature/*` PR.

## 2. Pull Request Policy

**Before opening the PR**
- Rebase onto the latest `main`:
  ```
  git fetch origin
  git rebase origin/main
  ```
  Do **not** merge `main` back into your feature branch — keeps history linear.
- Run `ruff`, `black`, and `pytest` locally inside the dev container (`make shell`). PR will be blocked by CI otherwise.

**Reviewers**
- At least **one approving review** from a member who is *not* the author.
- Any change that crosses module boundaries (e.g., A changes the dataloader output schema that B consumes) requires an approving review from **each affected module owner**.
- Author responds to or resolves every review comment before merging.

**PR description must contain**
1. **What changed** — one or two sentences.
2. **Why / which task** — reference to the [CLAUDE.md](../CLAUDE.md) task or proposal section it addresses.
3. **How it was tested** — exact command run inside `make shell`, or a link to a notebook cell with the result.

**Merging**
- **Squash-merge** into `main`. Each feature lands as one atomic commit on the trunk; the squashed commit message follows Conventional Commits (`feat(data): ...`, `fix(model): ...`).
- Branch is auto-deleted on merge.
- Direct pushes to `main` are forbidden — protect the branch on GitHub with "Require a pull request before merging" + "Require approvals = 1".

**Conflicts**
- Author owns the rebase. If the conflict touches another module, ping that owner on the PR before resolving so you don't silently overwrite their intent.

**Experiment branches (`exp/*`)**
- Exempt from review and CI gating.
- Must **not** be merged to `main`. Cherry-pick or re-author the winning change into a `feature/*` branch and open a normal PR.

## 3. `.gitignore`

The repository-root [.gitignore](../.gitignore) prevents accidentally committing model weights (`*.pth`, `*.safetensors`, `*.ckpt`, ...), dataset blobs (`data/raw/`, `data/processed/`, `*.h5`, `*.npy`, ...), experiment outputs (`wandb/`, `runs/`, `outputs/`), and Jupyter checkpoint caches. Add to it via a `chore/*` PR — never via a direct push.
