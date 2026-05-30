"""Predict Jianpu from staff image file(s) using a trained checkpoint.

Thin CLI over `src.deploy.OMRInferencer`. The encoder is inferred from the
checkpoint directory name (`vit-...` / `resnet-...`) unless `--encoder` is given.

Usage:
    python -m scripts.predict_jianpu \\
        --ckpt checkpoints/vit-20260528-090804 \\
        --image data/synthetic/val/000000.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.deploy import (
    OMRInferencer,
    jianpu_svg,
    pretty_jianpu,
    render_staff_png,
)


def _indexed(path: Path, i: int, multi: bool) -> Path:
    return path.with_name(f"{path.stem}-{i}{path.suffix}") if multi else path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument(
        "--encoder", default=None, help="vit/resnet; default: from ckpt name"
    )
    p.add_argument("--image", type=Path, nargs="+", required=True)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--format",
        choices=["compact", "pretty"],
        default="compact",
        help="stdout text: compact single-line ASCII (default, grep-friendly), "
        "or pretty 3-row monospace. For aligned dots/beams use --svg.",
    )
    p.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="write an aligned Jianpu SVG here (octave dots + rhythm beams). "
        "With multiple images, an index is appended per file.",
    )
    p.add_argument(
        "--engrave",
        type=Path,
        default=None,
        help="write an engraved staff PNG here (lilypond if installed, else "
        "verovio). With multiple images, an index is appended per file.",
    )
    args = p.parse_args()

    inferencer = OMRInferencer(
        args.ckpt,
        encoder=args.encoder,
        device=args.device,
        max_length=args.max_length,
    )
    print(
        f"[loaded] encoder={inferencer.encoder_name} "
        f"weights={inferencer.checkpoint_path} device={inferencer.device}"
    )

    preds = inferencer.predict_batch(list(args.image))
    multi = len(preds) > 1
    for i, (path, pred) in enumerate(zip(args.image, preds)):
        print(f"\n=== {path} (len={pred.length}) ===")
        print(pretty_jianpu(pred.tuples) if args.format == "pretty" else pred.jianpu)
        if args.svg is not None:
            out = _indexed(args.svg, i, multi)
            out.write_text(jianpu_svg(pred.tuples), encoding="utf-8")
            print(f"[svg] {out}")
        if args.engrave is not None:
            img, backend = render_staff_png(pred.tuples)
            out = _indexed(args.engrave, i, multi)
            Image.fromarray(img).save(out)
            print(f"[engraved:{backend}] {out}")


if __name__ == "__main__":
    main()
