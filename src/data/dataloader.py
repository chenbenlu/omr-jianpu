from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.augmentation import get_eval_augmentation, get_train_augmentation
from src.data.dataset import PrIMuSDataset
from src.data.splits import (
    discover_sample_ids,
    filter_overlong,
    find_corpus_root,
    split_sample_ids,
)
from src.data.vocabulary import Vocabulary

_logger = logging.getLogger(__name__)


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    label_lengths = torch.tensor(
        [item["label_length"] for item in batch], dtype=torch.long
    )
    max_len = int(label_lengths.max().item())

    padded: list[torch.Tensor] = []
    for item in batch:
        lbl: torch.Tensor = item["labels"]
        if lbl.shape[0] < max_len:
            pad = torch.zeros(max_len - lbl.shape[0], dtype=torch.long)
            lbl = torch.cat([lbl, pad])
        padded.append(lbl[:max_len])
    labels = torch.stack(padded)

    attention_mask = torch.zeros_like(labels)
    for i, length in enumerate(label_lengths):
        attention_mask[i, :length] = 1

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "attention_mask": attention_mask,
        "label_lengths": label_lengths,
    }


def create_dataloaders(
    data_dir: str | Path,
    vocab_path: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_camera: bool = True,
    max_seq_len: int = 512,
    image_height: int = 128,
    image_width: int = 1024,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    corpus_root = find_corpus_root(Path(data_dir))
    sample_ids = discover_sample_ids(corpus_root)
    if not sample_ids:
        raise ValueError(f"No valid PrIMuS samples found in {corpus_root}")

    # Filter before splitting so the seeded shuffle is over the kept set; this
    # keeps splits reproducible across runs with the same dataset on disk.
    # max_tokens = max_seq_len - 1 reserves one slot for EOS (no BOS in labels).
    max_tokens = max_seq_len - 1
    sample_ids, dropped = filter_overlong(corpus_root, sample_ids, max_tokens)
    if dropped:
        _logger.warning(
            "Filtered %d sample(s) with >%d semantic tokens "
            "(would not fit max_seq_len=%d). Examples: %s",
            len(dropped),
            max_tokens,
            max_seq_len,
            dropped[:5],
        )

    splits = split_sample_ids(sample_ids, train_ratio, val_ratio, seed)

    vocab_file = Path(vocab_path) if vocab_path else corpus_root / "vocab.json"
    train_semantic_files = [
        corpus_root / sid / f"{sid}.semantic" for sid in splits["train"]
    ]
    vocab = Vocabulary.load_or_build(vocab_file, train_semantic_files)

    train_transform = get_train_augmentation(height=image_height, width=image_width)
    eval_transform = get_eval_augmentation(height=image_height, width=image_width)

    datasets = {
        "train": PrIMuSDataset(
            corpus_root,
            vocab,
            splits["train"],
            train_transform,
            use_camera,
            max_seq_len,
        ),
        "val": PrIMuSDataset(
            corpus_root, vocab, splits["val"], eval_transform, use_camera, max_seq_len
        ),
        "test": PrIMuSDataset(
            corpus_root, vocab, splits["test"], eval_transform, use_camera, max_seq_len
        ),
    }

    loaders: dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        is_train = split == "train"
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )

    return loaders
