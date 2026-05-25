from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import PreRenderedOMRDataset, SyntheticOMRDataset
from src.data.encoders import EncoderSpec, get_encoder_spec
from src.data.generator import GeneratorConfig, MelodyGenerator
from src.data.renderer import RenderConfig, StaffRenderer
from src.data.vocabulary import (
    VocabBundle,
    Vocabulary,
    build_default_vocabs,
    load_bundle,
    save_bundle,
)

_LABEL_KEYS = ("type_ids", "pitch_ids", "rhythm_ids", "attribute_ids")


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("collate_fn received empty batch")

    pixel_values = _stack_or_pad_images(
        [item["pixel_values"] for item in batch], pad_value=1.0
    )

    label_lengths = torch.tensor(
        [int(item["label_length"]) for item in batch], dtype=torch.long
    )
    max_len = int(label_lengths.max().item())

    stacked: dict[str, torch.Tensor] = {}
    for key in _LABEL_KEYS:
        rows = []
        for item in batch:
            row: torch.Tensor = item[key]
            if row.shape[0] < max_len:
                row = torch.cat(
                    [
                        row,
                        torch.full(
                            (max_len - row.shape[0],),
                            Vocabulary.PAD_ID,
                            dtype=torch.long,
                        ),
                    ]
                )
            rows.append(row[:max_len])
        stacked[key] = torch.stack(rows)

    decoder_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, length in enumerate(label_lengths.tolist()):
        decoder_attention_mask[i, :length] = 1

    out: dict[str, torch.Tensor] = {
        "pixel_values": pixel_values,
        "decoder_attention_mask": decoder_attention_mask,
        "label_lengths": label_lengths,
    }
    out.update(stacked)
    return out


def _stack_or_pad_images(tensors: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    if not tensors:
        raise ValueError("no image tensors to stack")
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) == 1:
        return torch.stack(tensors)

    # Dynamic width: agree on (C, H), pad along W to batch-max.
    heights = {t.shape[-2] for t in tensors}
    channels = {t.shape[0] for t in tensors}
    if len(heights) != 1 or len(channels) != 1:
        raise ValueError(
            f"variable channels/heights in batch: shapes={shapes}; "
            "only width can vary"
        )
    c = next(iter(channels))
    h = next(iter(heights))
    max_w = max(t.shape[-1] for t in tensors)
    out = torch.full(
        (len(tensors), c, h, max_w), fill_value=pad_value, dtype=tensors[0].dtype
    )
    for i, t in enumerate(tensors):
        out[i, :, :, : t.shape[-1]] = t
    return out


def create_dataloaders(
    out_dir: str | Path,
    encoder: str | EncoderSpec = "vit",
    train_size: int = 10_000,
    val_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    max_seq_len: int = 512,
    generator_config: GeneratorConfig | None = None,
    render_config: RenderConfig | None = None,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    spec = get_encoder_spec(encoder)
    out_dir = Path(out_dir)

    vocab_dir = out_dir / "vocab"
    if (vocab_dir / "type.json").exists():
        vocabs = load_bundle(vocab_dir)
    else:
        vocabs = build_default_vocabs()
        save_bundle(vocabs, vocab_dir)

    generator = MelodyGenerator(generator_config)
    renderer = StaffRenderer(render_config)
    train_transform = spec.build_train_transform()
    eval_transform = spec.build_eval_transform()

    train_ds: Dataset = SyntheticOMRDataset(
        generator=generator,
        renderer=renderer,
        vocabs=vocabs,
        encoder_spec=spec,
        transform=train_transform,
        length=train_size,
        seed=seed,
        max_seq_len=max_seq_len,
    )

    val_ds = _build_eval_dataset(
        val_dir,
        vocabs,
        eval_transform,
        max_seq_len,
        generator,
        renderer,
        spec,
        seed + 10**6,
    )
    test_ds = _build_eval_dataset(
        test_dir,
        vocabs,
        eval_transform,
        max_seq_len,
        generator,
        renderer,
        spec,
        seed + 2 * 10**6,
    )

    return {
        "train": _make_loader(
            train_ds, batch_size, num_workers, pin_memory, shuffle=True
        ),
        "val": _make_loader(val_ds, batch_size, num_workers, pin_memory, shuffle=False),
        "test": _make_loader(
            test_ds, batch_size, num_workers, pin_memory, shuffle=False
        ),
    }


def _build_eval_dataset(
    split_dir: str | Path | None,
    vocabs: VocabBundle,
    transform: Any,
    max_seq_len: int,
    generator: MelodyGenerator,
    renderer: StaffRenderer,
    spec: EncoderSpec,
    fallback_seed: int,
) -> Dataset:
    if split_dir is None:
        return SyntheticOMRDataset(
            generator=generator,
            renderer=renderer,
            vocabs=vocabs,
            encoder_spec=spec,
            transform=transform,
            length=max(1, min(256, 256)),
            seed=fallback_seed,
            max_seq_len=max_seq_len,
        )
    manifest = Path(split_dir) / "manifest.jsonl"
    return PreRenderedOMRDataset(manifest, vocabs, transform, max_seq_len)


def _make_loader(
    ds: Dataset, batch_size: int, num_workers: int, pin_memory: bool, *, shuffle: bool
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )
