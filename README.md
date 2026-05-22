# OMR-to-Jianpu

End-to-end deep-learning Optical Music Recognition. A Vision-Encoder-Decoder translates printed staff-notation images directly into [Jianpu](https://en.wikipedia.org/wiki/Numbered_musical_notation) (numbered notation) semantic tokens — no MusicXML, no MIDI, no rule-based heuristics.

NYCU 535354 Deep Learning final project (Track 3 — Application). Team: BEN-LU CHEN, CHUN-JUI HSU, MENG-XI LIN, JIAN-AN ZHU. See [docs/proposal/proposal.pdf](docs/proposal/proposal.pdf) for the full proposal.

## Quick start

You need WSL2 + Docker Desktop with the NVIDIA Container Toolkit, on a machine with a Blackwell GPU (RTX 5060 / 5070 / 6000).

```bash
make build       # ~10 min first time (pulls ~6 GB CUDA base + installs deps)
make up          # start the dev container in the background
make shell       # drop into a bash shell as the non-root `omr` user
```

Sanity checks:

```bash
make gpu         # nvidia-smi inside the container
make gpu-test    # 2048×2048 GPU matmul — confirms sm_120 kernels are present
make image-size  # should be ~30 GB; if much larger, see the chown trap note below
```

When you're done:

```bash
make down        # stop and remove the container (your code on the host is untouched)
```

## Why these specific versions

The Dockerfile pins `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. **Do not downgrade.** All three team GPUs are Blackwell (compute capability `sm_120`), and PyTorch wheels older than 2.7 ship kernels only up to `sm_90`. They run `nvidia-smi` fine and `torch.cuda.is_available()` returns `True`, then every actual GPU op crashes with:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

If `make build` ever produces a multi-tens-of-GB image, the most likely cause is a `chown -R` over `/opt/conda` — Docker's copy-on-write duplicates every chowned file into a new layer. The current Dockerfile installs deps at build time as root and never recursively chowns parent-layer dirs; keep it that way.

## Repository layout

```
.
├── Dockerfile, docker-compose.yml, Makefile     # dev environment
├── requirements.txt                              # Python deps (no torch — comes from base image)
├── .dockerignore, .gitignore                     # keep build context small; keep repo clean
├── .pre-commit-config.yaml                       # ruff + black + nbstripout + large-file guard
├── .github/
│   ├── workflows/ci.yml                          # ruff + black + pytest + branch-name check
│   └── PULL_REQUEST_TEMPLATE.md
├── src/
│   ├── data/         # Member A — Camera-PrIMuS parsing, augmentation, DataLoaders
│   ├── model/        # Member B — Vision-Encoder-Decoder, losses, training loop
│   ├── postproc/     # Member C — semantic-token → Jianpu mapping
│   └── deploy/       # Member D — pipeline glue + Streamlit UI
├── configs/          # Hydra configs land here
├── data/             # gitignored — datasets live here at runtime
├── notebooks/        # exploratory work; not on the import path
├── tests/
└── docs/
    ├── GIT_WORKFLOW.md      # branching, PR policy, conflict resolution
    └── proposal/            # LaTeX source + rendered PDF for the NeurIPS-style proposal
```

## Collaboration

We use a lightweight GitHub Flow split by module owner. The full policy lives in [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md); the short version:

- Branch names: `<type>/<owner-letter>-<kebab-slug>` — e.g. `feature/B-train-vit-decoder-baseline`, `fix/A-augment-rotation-clipping`, `exp/B-cosine-lr-warmup-trial`.
- `main` is protected. PRs need ≥ 1 approving review (every affected module owner if the change crosses module lines). Squash-merge into `main`. Rebase your branch onto `main`; never merge `main` back.
- Experiments live on `exp/*` branches and **must not** be merged to `main`. Re-author the winning idea as a `feature/*` PR.

Before your first commit:

```bash
make shell
pre-commit install   # one-time, inside the container
```

## Documentation index

- [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) — full Git workflow policy
- [docs/proposal/proposal.pdf](docs/proposal/proposal.pdf) — project proposal
