"""Parallel pre-render of a synthetic OMR split to disk.

The on-the-fly `SyntheticOMRDataset` re-renders every image with verovio on
each epoch, which is CPU-bound and starves the GPU. Pre-rendering the train
split once lets training read PNGs from disk (augmentation still applied at
load time), making the run GPU-bound. Sharded across processes for speed.

Usage:
    python -m scripts.prerender_train --out data/synthetic/train \\
        --n 100000 --seed 42 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from src.data.generator import MelodyGenerator
from src.data.renderer import StaffRenderer

_LABEL_KEYS = ("type", "pitch", "rhythm", "attribute")

# Per-worker generator/renderer (verovio toolkit is rebuilt per process).
_gen: MelodyGenerator | None = None
_ren: StaffRenderer | None = None
_out: str = ""
_seed: int = 0


def _init(out_dir: str, seed: int) -> None:
    global _gen, _ren, _out, _seed
    _gen = MelodyGenerator()
    _ren = StaffRenderer()
    _out = out_dir
    _seed = seed


def _render_one(idx: int) -> tuple[int, dict]:
    assert _gen is not None and _ren is not None
    sample = _gen.generate(_seed, idx)
    image = _ren.render(sample.stream)
    img_name = f"{idx:06d}.png"
    Image.fromarray(image).save(os.path.join(_out, img_name))
    record: dict = {"image": img_name}
    for key in _LABEL_KEYS:
        record[key] = sample.labels[key]
    return idx, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records: dict[int, dict] = {}

    from tqdm import tqdm

    with Pool(
        args.workers, initializer=_init, initargs=(str(args.out), args.seed)
    ) as pool:
        for idx, rec in tqdm(
            pool.imap_unordered(_render_one, range(args.n), chunksize=64),
            total=args.n,
            desc=f"render {args.out.name}",
        ):
            records[idx] = rec

    manifest = args.out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for i in range(args.n):
            fh.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    print(f"wrote {args.n} samples -> {manifest}")


if __name__ == "__main__":
    main()
