from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.data.generator import MelodyGenerator
from src.data.renderer import StaffRenderer

_LABEL_KEYS = ("type", "pitch", "rhythm", "attribute")


def prerender_split(
    out_dir: str | Path,
    generator: MelodyGenerator,
    renderer: StaffRenderer,
    n: int,
    seed: int,
    progress: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    iterator: Iterable[int] = range(n)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, total=n, desc=f"render {out_dir.name}")

    with manifest_path.open("w", encoding="utf-8") as fh:
        for idx in iterator:
            sample = generator.generate(seed, idx)
            image = renderer.render(sample.stream)
            img_name = f"{idx:06d}.png"
            Image.fromarray(image).save(out_dir / img_name)
            record = {"image": img_name}
            for key in _LABEL_KEYS:
                record[key] = sample.labels[key]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pre-render a synthetic OMR split")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--n", type=int, required=True, help="number of samples")
    parser.add_argument("--seed", type=int, required=True, help="generator seed")
    parser.add_argument(
        "--split", type=str, default="val", help="label only — for logs"
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    generator = MelodyGenerator()
    renderer = StaffRenderer()
    manifest = prerender_split(
        args.out,
        generator,
        renderer,
        n=args.n,
        seed=args.seed,
        progress=not args.no_progress,
    )
    print(f"Wrote {args.n} samples for split={args.split} -> {manifest}")


if __name__ == "__main__":
    _main()
