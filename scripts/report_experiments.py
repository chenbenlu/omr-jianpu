"""Turn TensorBoard training logs into paper-ready figures + a summary table.

Auto-discovers encoder variants from the run-dir names (`<encoder>-<timestamp>`)
under `runs/`, groups runs by encoder, and overlays them. When a variant has
several runs (e.g. the encoder-ablation seeds) the mean is drawn with a
min/max band. No external dependency beyond `tensorboard` + `matplotlib`.

Usage:
    python -m scripts.report_experiments --runs-dir runs --out reports/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

# (tag, title, ylabel, log_y) for each figure.
METRICS = [
    ("val/ser", "Validation Symbol Error Rate", "SER (lower is better)", True),
    ("val/pitch_accuracy", "Validation Pitch Accuracy", "accuracy", False),
    ("val/rhythm_accuracy", "Validation Rhythm Accuracy", "accuracy", False),
    ("train/loss/total", "Training Loss", "total loss", True),
]


@dataclass
class RunCurves:
    encoder: str
    name: str
    scalars: dict[str, tuple[list[int], list[float]]]


def _encoder_of(run_name: str) -> str:
    return run_name.split("-", 1)[0]


def load_runs(runs_dir: Path) -> list[RunCurves]:
    runs: list[RunCurves] = []
    for run_dir in sorted(p for p in runs_dir.glob("*") if p.is_dir()):
        if not any(run_dir.glob("events.out.tfevents.*")):
            continue
        ea = EventAccumulator(str(run_dir))
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        scalars: dict[str, tuple[list[int], list[float]]] = {}
        for tag, *_ in METRICS:
            if tag in available:
                events = ea.Scalars(tag)
                scalars[tag] = ([e.step for e in events], [e.value for e in events])
        if scalars:
            runs.append(RunCurves(_encoder_of(run_dir.name), run_dir.name, scalars))
    return runs


def plot_metric(
    runs: list[RunCurves],
    tag: str,
    title: str,
    ylabel: str,
    log_y: bool,
    out_path: Path,
) -> bool:
    """One line per run, colored consistently per encoder.

    Runs of the same encoder (e.g. ablation seeds / restarts) log at different
    step cadences, so they are NOT averaged onto a forced common grid — each
    run is drawn on its own steps. The encoder appears once in the legend; the
    shared color makes the encoder-vs-encoder separation read at a glance.
    """
    by_encoder: dict[str, list[RunCurves]] = defaultdict(list)
    for run in runs:
        if tag in run.scalars:
            by_encoder[run.encoder].append(run)
    if not by_encoder:
        return False

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for color_idx, encoder in enumerate(sorted(by_encoder)):
        color = cmap(color_idx)
        for first, run in enumerate(by_encoder[encoder]):
            steps, values = run.scalars[tag]
            n = len(by_encoder[encoder])
            label = (f"{encoder} (n={n})" if n > 1 else encoder) if first == 0 else None
            ax.plot(steps, values, color=color, linewidth=1.8, alpha=0.85, label=label)

    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def write_summary(runs: list[RunCurves], out_dir: Path) -> list[dict]:
    """Best (min) val/SER and final pitch/rhythm accuracy, per run."""
    rows: list[dict] = []
    for run in runs:
        row = {"run": run.name, "encoder": run.encoder}
        if "val/ser" in run.scalars:
            row["best_val_ser"] = round(min(run.scalars["val/ser"][1]), 5)
        for tag, key in [
            ("val/pitch_accuracy", "final_pitch_acc"),
            ("val/rhythm_accuracy", "final_rhythm_acc"),
        ]:
            if tag in run.scalars:
                row[key] = round(run.scalars[tag][1][-1], 5)
        rows.append(row)

    fields = ["run", "encoder", "best_val_ser", "final_pitch_acc", "final_rhythm_acc"]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--format", default="png", choices=["png", "pdf"])
    args = p.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no runs with scalar logs found under {args.runs_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    for tag, title, ylabel, log_y in METRICS:
        fname = tag.replace("/", "_") + "." + args.format
        if plot_metric(runs, tag, title, ylabel, log_y, args.out / fname):
            print(f"[fig] {args.out / fname}")

    rows = write_summary(runs, args.out)
    encoders = sorted({r.encoder for r in runs})
    print(
        f"[summary] {len(rows)} runs, encoders={encoders} -> "
        f"{args.out / 'summary.csv'}, {args.out / 'summary.md'}"
    )


if __name__ == "__main__":
    main()
