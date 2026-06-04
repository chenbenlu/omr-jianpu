"""Compare a baseline vs a fine-tuned OMR checkpoint on the real photographed
scores in ``data/external/``.

The external photos are phone captures of the same deterministic synthetic
samples used for the held-out splits (the zips are named by seed 1000000 / val
and 2000000 / test), so each image ``NNNNNN.png`` pairs 1:1 *by filename* with a
ground-truth row in ``data/synthetic/{val,test}/manifest.jsonl``. That lets us
score SER / pitch-accuracy / rhythm-accuracy quantitatively instead of only
eyeballing predictions.

Outputs (under ``--out``):
  * ``summary.md`` / ``summary.csv`` — metrics per (model x split) and combined.
  * ``predictions.jsonl`` — per image: GT tuples + each model's predicted tuples,
    Jianpu and per-sample SER, for side-by-side inspection.

Usage:
    python -m scripts.eval_external \
        --baseline checkpoints/vit-20260528-090804 \
        --finetuned checkpoints/vit-photoft-001 \
        --out reports/external_compare
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from PIL import Image

from src.deploy.inference import JianpuPrediction, OMRInferencer
from src.postproc import EvalMetrics, TokenTuple, aggregate, evaluate

# Each external zip -> its label manifest. Zip filenames carry spaces/parens.
_SPLITS: dict[str, dict[str, str]] = {
    "val": {
        "zip": "val (seed 1000000)_picture.zip",
        "manifest": "data/synthetic/val/manifest.jsonl",
    },
    "test": {
        "zip": "test (seed 2000000)_picture.zip",
        "manifest": "data/synthetic/test/manifest.jsonl",
    },
}


def _load_manifest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["image"]] = row
    return rows


def _gt_tuples(row: dict) -> list[TokenTuple]:
    """Build ground-truth tuples straight from a manifest row.

    The four parallel lists are position-aligned; JSON ``null`` -> ``None``,
    which matches what ``ids_to_tuples`` emits for ``<NULL>`` heads, so equality
    against predictions is exact.
    """
    return [
        (t, p, r, a)
        for t, p, r, a in zip(
            row["type"], row["pitch"], row["rhythm"], row["attribute"]
        )
    ]


def _read_zip_images(zip_path: Path) -> list[tuple[str, Image.Image]]:
    out: list[tuple[str, Image.Image]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
        for n in names:
            img = Image.open(io.BytesIO(zf.read(n))).convert("RGB")
            out.append((Path(n).name, img))
    return out


def _chunks(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _predict_all(
    inf: OMRInferencer, images: list[Image.Image], batch_size: int
) -> list[JianpuPrediction]:
    preds: list[JianpuPrediction] = []
    for chunk in _chunks(images, batch_size):
        preds.extend(inf.predict_batch(chunk))
    return preds


def _fmt(m: EvalMetrics, n: int) -> dict[str, str]:
    return {
        "n": str(n),
        "SER": f"{m.ser:.4f}",
        "pitch_acc": f"{m.pitch_accuracy:.4f}",
        "rhythm_acc": f"{m.rhythm_accuracy:.4f}",
        "edit_dist": str(m.edit_distance),
        "gt_len": str(m.gt_length),
    }


_COLS = ("model", "split", "n", "SER", "pitch_acc", "rhythm_acc", "edit_dist", "gt_len")


def _markdown_table(rows: list[dict[str, str]]) -> str:
    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    body = ["| " + " | ".join(r.get(c, "") for c in _COLS) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _csv_table(rows: list[dict[str, str]]) -> str:
    lines = [",".join(_COLS)]
    lines += [",".join(r.get(c, "") for c in _COLS) for r in rows]
    return "\n".join(lines) + "\n"


def run(
    baseline: str,
    finetuned: str | None,
    external_dir: Path,
    out_dir: Path,
    batch_size: int,
    max_length: int,
) -> None:
    models: dict[str, OMRInferencer] = {
        "baseline": OMRInferencer(baseline, encoder="vit", max_length=max_length),
    }
    if finetuned:
        models["finetuned"] = OMRInferencer(
            finetuned, encoder="vit", max_length=max_length
        )
    for name, inf in models.items():
        print(f"[{name}] loaded checkpoint: {inf.checkpoint_path}")

    # per (model, split) -> per-sample metrics; dumps keyed by (split, image).
    per: dict[tuple[str, str], list[EvalMetrics]] = {}
    dumps: dict[tuple[str, str], dict] = {}

    for split, info in _SPLITS.items():
        manifest = _load_manifest(Path(info["manifest"]))
        images = _read_zip_images(external_dir / info["zip"])
        labeled = [(n, img) for n, img in images if n in manifest]
        skipped = len(images) - len(labeled)
        print(
            f"[{split}] {len(labeled)} labeled photos"
            + (f" ({skipped} without a manifest match skipped)" if skipped else "")
        )
        names = [n for n, _ in labeled]
        pil = [img for _, img in labeled]
        gts = [_gt_tuples(manifest[n]) for n in names]

        for n, gt in zip(names, gts):
            dumps.setdefault(
                (split, n),
                {"split": split, "image": n, "gt": [list(t) for t in gt]},
            )

        for model_name, inf in models.items():
            preds = _predict_all(inf, pil, batch_size)
            sample_metrics: list[EvalMetrics] = []
            for n, gt, pred in zip(names, gts, preds):
                m = evaluate(gt, pred.tuples)
                sample_metrics.append(m)
                dumps[(split, n)][model_name] = {
                    "ser": round(m.ser, 4),
                    "jianpu": pred.jianpu,
                    "tuples": [list(t) for t in pred.tuples],
                }
            per[(model_name, split)] = sample_metrics

    # Build summary rows: per (model, split) + combined across splits.
    table_rows: list[dict[str, str]] = []
    for model_name in models:
        combined: list[EvalMetrics] = []
        for split in _SPLITS:
            sm = per[(model_name, split)]
            combined += sm
            table_rows.append(
                {"model": model_name, "split": split, **_fmt(aggregate(sm), len(sm))}
            )
        table_rows.append(
            {
                "model": model_name,
                "split": "all",
                **_fmt(aggregate(combined), len(combined)),
            }
        )

    md = _markdown_table(table_rows)
    print("\n" + md + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    title = "# External (photographed) evaluation: baseline vs fine-tuned\n\n"
    (out_dir / "summary.md").write_text(title + md + "\n")
    (out_dir / "summary.csv").write_text(_csv_table(table_rows))
    with (out_dir / "predictions.jsonl").open("w") as fh:
        for key in sorted(dumps):
            fh.write(json.dumps(dumps[key], ensure_ascii=False) + "\n")
    print(f"wrote {out_dir/'summary.md'}, summary.csv, predictions.jsonl")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline",
        default="checkpoints/vit-20260528-090804",
        help="baseline (un-fine-tuned) checkpoint dir or run dir",
    )
    ap.add_argument(
        "--finetuned",
        default=None,
        help="fine-tuned checkpoint dir; omit to score the baseline alone",
    )
    ap.add_argument("--external-dir", default="data/external", type=Path)
    ap.add_argument(
        "--out", dest="out_dir", default="reports/external_compare", type=Path
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64)
    return ap.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = _parse_args()
    run(
        baseline=args.baseline,
        finetuned=args.finetuned,
        external_dir=args.external_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
