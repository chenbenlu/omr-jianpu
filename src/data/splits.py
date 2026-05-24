from __future__ import annotations

import random
from pathlib import Path


def find_corpus_root(data_dir: Path) -> Path:
    # Camera-PrIMuS tarball (see src/data/download.py) extracts samples into
    # `<data_dir>/Corpus/`. Tests use a flat layout, so fall back to data_dir.
    data_dir = Path(data_dir)
    corpus = data_dir / "Corpus"
    return corpus if corpus.is_dir() else data_dir


def discover_sample_ids(corpus_root: Path) -> list[str]:
    corpus_root = Path(corpus_root)
    return sorted(
        d.name
        for d in corpus_root.iterdir()
        if d.is_dir() and (d / f"{d.name}.semantic").exists()
    )


def split_sample_ids(
    sample_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    ids = list(sample_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def filter_overlong(
    corpus_root: Path,
    sample_ids: list[str],
    max_tokens: int,
) -> tuple[list[str], list[str]]:
    # Reads each {sid}.semantic, counts tab-separated tokens, partitions into
    # (kept, dropped) where kept has token_count <= max_tokens. Pure; no logging.
    corpus_root = Path(corpus_root)
    kept: list[str] = []
    dropped: list[str] = []
    for sid in sample_ids:
        semantic_path = corpus_root / sid / f"{sid}.semantic"
        text = semantic_path.read_text(encoding="utf-8").strip()
        n_tokens = sum(1 for t in text.split("\t") if t.strip())
        (kept if n_tokens <= max_tokens else dropped).append(sid)
    return kept, dropped
