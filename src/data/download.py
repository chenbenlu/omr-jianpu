"""Download the Camera-PrIMuS dataset.

Usage:
    python -m src.data.download
    python -m src.data.download --dest data/raw/primus
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

CAMERA_PRIMUS_URL = "https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz"


class _TqdmHook(tqdm):
    def update_to(
        self, blocks: int = 1, block_size: int = 1, total_size: int | None = None
    ) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_primus(dest: Path, url: str = CAMERA_PRIMUS_URL) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    existing = [d for d in dest.iterdir() if d.is_dir()]
    if existing:
        print(
            f"Dataset already extracted at {dest} ({len(existing)} samples). Skipping download."
        )
        return

    print(f"Downloading Camera-PrIMuS from {url}")
    with tempfile.NamedTemporaryFile(suffix=".tgz", dir=dest, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with _TqdmHook(
            unit="B", unit_scale=True, miniters=1, desc="CameraPrIMuS.tgz"
        ) as t:
            urllib.request.urlretrieve(url, tmp_path, reporthook=t.update_to)

        print(f"Extracting to {dest} …")
        with tarfile.open(tmp_path, "r:gz") as tf:
            tf.extractall(dest)
        print("Done.")
    finally:
        tmp_path.unlink(missing_ok=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Download Camera-PrIMuS dataset")
    parser.add_argument(
        "--dest",
        default="data/raw/primus",
        help="Destination directory (default: data/raw/primus)",
    )
    parser.add_argument(
        "--url", default=CAMERA_PRIMUS_URL, help="Override download URL"
    )
    args = parser.parse_args()
    download_primus(Path(args.dest), url=args.url)


if __name__ == "__main__":
    _main()
