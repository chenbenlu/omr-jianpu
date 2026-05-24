from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.augmentation import get_eval_augmentation, get_train_augmentation
from src.data.dataloader import collate_fn, create_dataloaders
from src.data.dataset import PrIMuSDataset
from src.data.splits import (
    discover_sample_ids,
    filter_overlong,
    find_corpus_root,
    split_sample_ids,
)
from src.data.vocabulary import Vocabulary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SEMANTIC = "note-C4_eighth\tbarline\tclef-G2\ttimesig-4_4\tnote-D4_quarter"
_TOKENS = ["note-C4_eighth", "barline", "clef-G2", "timesig-4_4", "note-D4_quarter"]


def _make_sample(root: Path, sid: str, semantic: str = _SEMANTIC) -> None:
    sample_dir = root / sid
    sample_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (256, 64), color=255)
    img.save(sample_dir / f"{sid}.png")
    img.save(sample_dir / f"{sid}.jpg")
    (sample_dir / f"{sid}.semantic").write_text(semantic, encoding="utf-8")


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "primus"
    for i in range(12):
        _make_sample(root, f"sample_{i:03d}")
    return root


@pytest.fixture()
def built_vocab(tiny_dataset: Path) -> Vocabulary:
    files = [
        tiny_dataset / f"sample_{i:03d}" / f"sample_{i:03d}.semantic" for i in range(12)
    ]
    return Vocabulary.build(files)


# ---------------------------------------------------------------------------
# Vocabulary tests
# ---------------------------------------------------------------------------


def test_vocabulary_build(built_vocab: Vocabulary) -> None:
    assert len(built_vocab) == len(Vocabulary.SPECIAL_TOKENS) + len(_TOKENS)
    assert built_vocab.token_to_id["<PAD>"] == Vocabulary.PAD_ID
    assert built_vocab.token_to_id["<BOS>"] == Vocabulary.BOS_ID
    assert built_vocab.token_to_id["<EOS>"] == Vocabulary.EOS_ID
    assert built_vocab.token_to_id["<UNK>"] == Vocabulary.UNK_ID
    # Music tokens start at 4
    for tok in _TOKENS:
        assert built_vocab.token_to_id[tok] >= 4


def test_vocabulary_encode_decode_roundtrip(built_vocab: Vocabulary) -> None:
    tokens = ["note-C4_eighth", "barline"]
    ids = built_vocab.encode(tokens)
    recovered = built_vocab.decode(ids)
    assert recovered == tokens


def test_vocabulary_decode_skip_special_tokens(built_vocab: Vocabulary) -> None:
    music_ids = built_vocab.encode(["note-C4_eighth", "barline"])
    framed = [Vocabulary.BOS_ID, *music_ids, Vocabulary.EOS_ID, Vocabulary.PAD_ID]
    assert built_vocab.decode(framed) == [
        "<BOS>",
        "note-C4_eighth",
        "barline",
        "<EOS>",
        "<PAD>",
    ]
    assert built_vocab.decode(framed, skip_special_tokens=True) == [
        "note-C4_eighth",
        "barline",
    ]
    # UNK survives — it is a real prediction, not a structural token.
    with_unk = [Vocabulary.BOS_ID, Vocabulary.UNK_ID, *music_ids, Vocabulary.EOS_ID]
    assert built_vocab.decode(with_unk, skip_special_tokens=True) == [
        "<UNK>",
        "note-C4_eighth",
        "barline",
    ]


def test_vocabulary_unk(built_vocab: Vocabulary) -> None:
    ids = built_vocab.encode(["NONEXISTENT_TOKEN"])
    assert ids[0] == Vocabulary.UNK_ID


def test_vocabulary_save_load(built_vocab: Vocabulary, tmp_path: Path) -> None:
    save_path = tmp_path / "vocab.json"
    built_vocab.save(save_path)
    loaded = Vocabulary.load(save_path)
    assert loaded.token_to_id == built_vocab.token_to_id
    assert len(loaded) == len(built_vocab)


def test_vocabulary_load_or_build(tiny_dataset: Path, tmp_path: Path) -> None:
    files = [
        tiny_dataset / f"sample_{i:03d}" / f"sample_{i:03d}.semantic" for i in range(12)
    ]
    vocab_path = tmp_path / "vocab.json"
    assert not vocab_path.exists()

    built = Vocabulary.load_or_build(vocab_path, files)
    assert vocab_path.exists()
    assert all(tok in built.token_to_id for tok in _TOKENS)

    # Second call must short-circuit to the cached file (no rebuild).
    cached = Vocabulary.load_or_build(vocab_path, [])
    assert cached.token_to_id == built.token_to_id


# ---------------------------------------------------------------------------
# Splits tests
# ---------------------------------------------------------------------------


def test_find_corpus_root_flat(tmp_path: Path) -> None:
    # No Corpus/ subdir → return data_dir unchanged.
    assert find_corpus_root(tmp_path) == tmp_path


def test_find_corpus_root_with_corpus_subdir(tmp_path: Path) -> None:
    corpus = tmp_path / "Corpus"
    corpus.mkdir()
    assert find_corpus_root(tmp_path) == corpus


def test_discover_sample_ids(tiny_dataset: Path) -> None:
    ids = discover_sample_ids(tiny_dataset)
    assert ids == sorted(f"sample_{i:03d}" for i in range(12))


def test_split_sample_ids_seeded() -> None:
    ids = [f"s{i:03d}" for i in range(100)]
    a = split_sample_ids(ids, train_ratio=0.8, val_ratio=0.1, seed=42)
    b = split_sample_ids(ids, train_ratio=0.8, val_ratio=0.1, seed=42)
    assert a == b
    assert len(a["train"]) == 80
    assert len(a["val"]) == 10
    assert len(a["test"]) == 10
    # Disjoint and covering
    union = set(a["train"]) | set(a["val"]) | set(a["test"])
    assert union == set(ids)
    # Different seed → different partition
    c = split_sample_ids(ids, train_ratio=0.8, val_ratio=0.1, seed=7)
    assert c["train"] != a["train"]


def test_filter_overlong(tmp_path: Path) -> None:
    root = tmp_path / "primus"
    counts = {"short": 2, "mid": 5, "long": 10}
    for sid, n in counts.items():
        sample_dir = root / sid
        sample_dir.mkdir(parents=True)
        (sample_dir / f"{sid}.semantic").write_text(
            "\t".join(f"tok{i}" for i in range(n)), encoding="utf-8"
        )

    kept, dropped = filter_overlong(root, list(counts), max_tokens=5)
    assert set(kept) == {"short", "mid"}
    assert dropped == ["long"]


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


def test_dataset_getitem(tiny_dataset: Path, built_vocab: Vocabulary) -> None:
    ids = [f"sample_{i:03d}" for i in range(12)]
    ds = PrIMuSDataset(
        tiny_dataset, built_vocab, ids, transform=get_eval_augmentation()
    )
    item = ds[0]
    assert item["pixel_values"].shape == (3, 128, 1024)
    assert item["pixel_values"].dtype == torch.float32
    labels: torch.Tensor = item["labels"]
    length: int = item["label_length"]
    # Labels are `tokens + EOS + PAD…` — no BOS. HF VED prepends BOS itself.
    assert labels[0].item() != Vocabulary.BOS_ID
    assert labels[0].item() == built_vocab.token_to_id[_TOKENS[0]]
    assert labels[length - 1].item() == Vocabulary.EOS_ID
    assert length == len(_TOKENS) + 1  # tokens + EOS
    assert labels[length:].eq(Vocabulary.PAD_ID).all()


def test_dataset_camera_fallback(tiny_dataset: Path, built_vocab: Vocabulary) -> None:
    # Remove .jpg for sample_000; should fall back to .png silently
    (tiny_dataset / "sample_000" / "sample_000.jpg").unlink()
    ids = ["sample_000"]
    ds = PrIMuSDataset(tiny_dataset, built_vocab, ids, use_camera=True)
    item = ds[0]
    assert item["pixel_values"].shape == (3, 128, 1024)


def test_dataset_label_truncation(tiny_dataset: Path, built_vocab: Vocabulary) -> None:
    ids = [f"sample_{i:03d}" for i in range(12)]
    ds = PrIMuSDataset(tiny_dataset, built_vocab, ids, max_seq_len=5)
    item = ds[0]
    assert item["labels"].shape[0] == 5
    # EOS must still be present (last real token)
    length: int = item["label_length"]
    assert item["labels"][length - 1].item() == Vocabulary.EOS_ID


# ---------------------------------------------------------------------------
# collate_fn tests
# ---------------------------------------------------------------------------


def test_collate_fn_padding(tiny_dataset: Path, built_vocab: Vocabulary) -> None:
    ids = [f"sample_{i:03d}" for i in range(12)]
    ds_long = PrIMuSDataset(tiny_dataset, built_vocab, ids, max_seq_len=512)
    ds_short = PrIMuSDataset(tiny_dataset, built_vocab, ids, max_seq_len=5)

    item_long = ds_long[0]
    item_short = ds_short[0]

    batch = collate_fn([item_short, item_long])
    max_len = item_long["label_length"]
    assert batch["labels"].shape == (2, max_len)
    assert batch["decoder_attention_mask"].shape == (2, max_len)
    assert batch["label_lengths"].tolist() == [
        item_short["label_length"],
        item_long["label_length"],
    ]

    short_len = item_short["label_length"]
    # Positions after short EOS must be masked out
    assert batch["decoder_attention_mask"][0, short_len:].eq(0).all()


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------


def test_create_dataloaders(tiny_dataset: Path) -> None:
    loaders = create_dataloaders(
        tiny_dataset, batch_size=4, num_workers=0, pin_memory=False
    )
    assert set(loaders.keys()) == {"train", "val", "test"}
    batch = next(iter(loaders["train"]))
    assert batch["pixel_values"].ndim == 4
    assert batch["labels"].ndim == 2
    assert batch["decoder_attention_mask"].shape == batch["labels"].shape
    assert batch["label_lengths"].ndim == 1
    assert "attention_mask" not in batch  # renamed; guard against regression


def test_create_dataloaders_vocab_saved(tiny_dataset: Path) -> None:
    create_dataloaders(tiny_dataset, batch_size=4, num_workers=0, pin_memory=False)
    assert (tiny_dataset / "vocab.json").exists()


def test_create_dataloaders_filters_overlong(
    tiny_dataset: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    # Add a sample whose semantic far exceeds max_seq_len-1 = 7.
    huge_dir = tiny_dataset / "huge"
    huge_dir.mkdir()
    Image.new("RGB", (256, 64), color=255).save(huge_dir / "huge.png")
    Image.new("RGB", (256, 64), color=255).save(huge_dir / "huge.jpg")
    (huge_dir / "huge.semantic").write_text(
        "\t".join(f"tok{i}" for i in range(50)), encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="src.data.dataloader"):
        loaders = create_dataloaders(
            tiny_dataset,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            max_seq_len=8,
        )

    for loader in loaders.values():
        assert "huge" not in loader.dataset.sample_ids

    assert any("huge" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------


def test_train_augmentation_smoke() -> None:
    img = np.ones((64, 256, 3), dtype=np.uint8) * 200
    pipe = get_train_augmentation(height=128, width=1024)
    out = pipe(image=img)["image"]
    assert out.shape == (3, 128, 1024)
    assert out.dtype == torch.float32


def test_eval_augmentation_deterministic() -> None:
    img = np.ones((64, 256, 3), dtype=np.uint8) * 200
    pipe = get_eval_augmentation(height=128, width=1024)
    out1 = pipe(image=img)["image"]
    out2 = pipe(image=img)["image"]
    assert torch.allclose(out1, out2)


def test_padding_is_white_paper() -> None:
    # 64×256 gray content; LongestMaxSize scales 2× to 128×512, leaving cols
    # 512..1023 as pure padding. Pad fill=255 → normalized = +1.0.
    img = np.ones((64, 256, 3), dtype=np.uint8) * 200
    pipe = get_eval_augmentation(height=128, width=1024)
    out = pipe(image=img)["image"]
    pad_region = out[:, :, 700:]
    assert torch.allclose(pad_region, torch.ones_like(pad_region), atol=1e-5)
