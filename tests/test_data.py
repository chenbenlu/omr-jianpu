from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from music21 import chord, meter

from src.data import (
    ENCODER_REGISTRY,
    EncoderSpec,
    GeneratorConfig,
    MelodyGenerator,
    PreRenderedOMRDataset,
    StaffRenderer,
    SyntheticOMRDataset,
    Vocabulary,
    build_default_vocabs,
    collate_fn,
    create_dataloaders,
    get_encoder_spec,
    load_bundle,
    save_bundle,
)
from src.data.prerender import prerender_split

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_default_vocabs_special_ids() -> None:
    vb = build_default_vocabs()
    for name, vocab in vb:
        assert vocab.token_to_id["<PAD>"] == Vocabulary.PAD_ID
        assert vocab.token_to_id["<BOS>"] == Vocabulary.BOS_ID
        assert vocab.token_to_id["<EOS>"] == Vocabulary.EOS_ID
        assert vocab.token_to_id["<UNK>"] == Vocabulary.UNK_ID
        if vocab.has_null:
            assert vocab.token_to_id["<NULL>"] == Vocabulary.NULL_ID, name
        else:
            assert "<NULL>" not in vocab.token_to_id, name
    assert vb.type.has_null is False
    assert vb.pitch.has_null is True
    assert vb.rhythm.has_null is True
    assert vb.attribute.has_null is True


def test_type_vocab_rejects_none() -> None:
    vb = build_default_vocabs()
    with pytest.raises(ValueError, match="has no <NULL>"):
        vb.type.encode([None])


def test_vocab_encode_decode_roundtrip() -> None:
    vb = build_default_vocabs()
    tokens = ["note", "rest", "barline", "clef"]
    ids = vb.type.encode(tokens)
    assert vb.type.decode(ids) == tokens

    pitch_tokens = ["C4", None, "F#5", "Bb3"]
    pitch_ids = vb.pitch.encode(pitch_tokens)
    assert pitch_ids[1] == Vocabulary.NULL_ID
    assert vb.pitch.decode(pitch_ids) == pitch_tokens


def test_vocab_save_load_bundle(tmp_path: Path) -> None:
    vb = build_default_vocabs()
    save_bundle(vb, tmp_path / "vocabs")
    vb2 = load_bundle(tmp_path / "vocabs")
    assert vb.type.token_to_id == vb2.type.token_to_id
    assert vb.pitch.token_to_id == vb2.pitch.token_to_id
    assert vb.rhythm.token_to_id == vb2.rhythm.token_to_id
    assert vb.attribute.token_to_id == vb2.attribute.token_to_id


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def test_generator_determinism() -> None:
    g = MelodyGenerator()
    a = g.generate(seed=42, sample_idx=7)
    b = g.generate(seed=42, sample_idx=7)
    assert a.labels == b.labels
    c = g.generate(seed=42, sample_idx=8)
    assert a.labels != c.labels


def test_generator_monophony() -> None:
    g = MelodyGenerator()
    for idx in range(20):
        sample = g.generate(seed=1, sample_idx=idx)
        assert not any(isinstance(e, chord.Chord) for e in sample.stream.recurse()), idx


def test_generator_bar_durations() -> None:
    g = MelodyGenerator(
        GeneratorConfig(num_bars_range=(3, 3), time_signatures=("4/4",))
    )
    sample = g.generate(seed=123, sample_idx=0)
    ts = meter.TimeSignature("4/4")
    expected = float(ts.barDuration.quarterLength)

    # Walk through generated symbols (skip the 3 header tokens), split on barlines.
    types = sample.labels["type"][3:]
    rhythms = sample.labels["rhythm"][3:]
    bar_qlens = [0.0]
    for t, r in zip(types, rhythms):
        if t == "barline":
            bar_qlens.append(0.0)
        else:
            assert r is not None, (t, r)
            from music21 import duration

            bar_qlens[-1] += float(
                duration.Duration(
                    type=r[:-4] if r.endswith("_dot") else r,
                    dots=1 if r.endswith("_dot") else 0,
                ).quarterLength
            )
    for qlen in bar_qlens:
        assert abs(qlen - expected) < 1e-6


def test_generator_label_streams_aligned() -> None:
    g = MelodyGenerator()
    sample = g.generate(seed=5, sample_idx=2)
    n = len(sample.labels["type"])
    for key in ("pitch", "rhythm", "attribute"):
        assert len(sample.labels[key]) == n


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------


def test_encoder_spec_dispatch() -> None:
    vit = get_encoder_spec("vit")
    assert vit.channels == 3 and vit.target_height == 224 and vit.target_width == 224
    rn = get_encoder_spec("resnet")
    assert rn.channels == 1 and rn.target_height == 128 and rn.target_width is None
    with pytest.raises(KeyError, match="Unknown encoder"):
        get_encoder_spec("mamba")
    custom = EncoderSpec(
        name="custom",
        channels=1,
        target_height=64,
        target_width=64,
        max_width=64,
        normalize_mean=(0.5,),
        normalize_std=(0.5,),
    )
    assert get_encoder_spec(custom) is custom


def test_encoder_specs_registered() -> None:
    assert set(ENCODER_REGISTRY) == {"vit", "resnet"}


def test_vit_transform_fixed_shape() -> None:
    spec = get_encoder_spec("vit")
    pipe = spec.build_eval_transform()
    img = np.full((300, 1200, 3), 255, dtype=np.uint8)
    out = pipe(image=img)["image"]
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_resnet_transform_dynamic_width() -> None:
    spec = get_encoder_spec("resnet")
    pipe = spec.build_eval_transform()
    for w_in in (200, 600, 1200):
        out = pipe(image=np.full((400, w_in, 3), 200, dtype=np.uint8))["image"]
        assert out.shape[0] == 1 and out.shape[1] == 128
        assert out.shape[2] <= spec.max_width
        # Aspect-preserving: width = w_in * 128 / 400
        assert out.shape[2] == round(w_in * 128 / 400)


def test_resnet_transform_clips_overflow() -> None:
    spec = get_encoder_spec("resnet")
    pipe = spec.build_eval_transform()
    # 100h x 10000w → resize to 128×12800, then center-crop to max_width.
    out = pipe(image=np.full((100, 10_000, 3), 200, dtype=np.uint8))["image"]
    assert out.shape == (1, 128, spec.max_width)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _fake_item(pv: torch.Tensor, length: int) -> dict[str, torch.Tensor | int]:
    return {
        "pixel_values": pv,
        "type_ids": torch.tensor(
            [7, 8, 9] + [4] * (length - 4) + [2], dtype=torch.long
        ),
        "pitch_ids": torch.tensor(
            [4, 4, 4] + [42] * (length - 4) + [2], dtype=torch.long
        ),
        "rhythm_ids": torch.tensor(
            [4, 4, 4] + [6] * (length - 4) + [2], dtype=torch.long
        ),
        "attribute_ids": torch.tensor(
            [5, 12, 28] + [4] * (length - 4) + [2], dtype=torch.long
        ),
        "label_length": length,
    }


def test_collate_vit_fast_path() -> None:
    pv = torch.ones((3, 224, 224), dtype=torch.float32)
    items = [_fake_item(pv, length=8) for _ in range(3)]
    batch = collate_fn(items)
    assert batch["pixel_values"].shape == (3, 3, 224, 224)
    assert batch["type_ids"].shape == (3, 8)
    assert batch["decoder_attention_mask"].shape == (3, 8)
    assert batch["label_lengths"].tolist() == [8, 8, 8]


def test_collate_resnet_dynamic_width() -> None:
    items = []
    for w, length in [(300, 6), (500, 10), (400, 8)]:
        pv = torch.zeros((1, 128, w), dtype=torch.float32)
        items.append(_fake_item(pv, length=length))
    batch = collate_fn(items)
    assert batch["pixel_values"].shape == (3, 1, 128, 500)
    # First sample's pad region (w >= 300) must be +1.0 (normalized white).
    assert batch["pixel_values"][0, :, :, 300:].eq(1.0).all().item()
    assert batch["pixel_values"][2, :, :, 400:].eq(1.0).all().item()
    # Max label length = 10
    assert batch["type_ids"].shape == (3, 10)
    assert batch["label_lengths"].tolist() == [6, 10, 8]


def test_collate_label_mask_alignment() -> None:
    pv = torch.zeros((3, 224, 224), dtype=torch.float32)
    items = [_fake_item(pv, length=5), _fake_item(pv, length=8)]
    batch = collate_fn(items)
    assert batch["decoder_attention_mask"][0, :5].eq(1).all().item()
    assert batch["decoder_attention_mask"][0, 5:].eq(0).all().item()
    assert batch["decoder_attention_mask"][1, :8].eq(1).all().item()
    # PAD positions should be PAD_ID across all four streams
    for key in ("type_ids", "pitch_ids", "rhythm_ids", "attribute_ids"):
        assert batch[key][0, 5:].eq(Vocabulary.PAD_ID).all().item()


# ---------------------------------------------------------------------------
# Dataset (with mocked renderer)
# ---------------------------------------------------------------------------


def _mock_render(_self, _score):  # type: ignore[no-untyped-def]
    return np.full((300, 1200, 3), 255, dtype=np.uint8)


@pytest.fixture()
def mocked_renderer():
    with patch.object(StaffRenderer, "render", _mock_render):
        yield


@pytest.mark.usefixtures("mocked_renderer")
def test_synthetic_dataset_vit() -> None:
    vb = build_default_vocabs()
    spec = get_encoder_spec("vit")
    ds = SyntheticOMRDataset(
        generator=MelodyGenerator(),
        renderer=StaffRenderer(),
        vocabs=vb,
        encoder_spec=spec,
        transform=spec.build_eval_transform(),
        length=4,
        seed=42,
        max_seq_len=128,
    )
    item = ds[0]
    assert item["pixel_values"].shape == (3, 224, 224)
    assert item["type_ids"].shape == (128,)
    assert item["pitch_ids"].shape == (128,)
    assert item["rhythm_ids"].shape == (128,)
    assert item["attribute_ids"].shape == (128,)
    L = item["label_length"]
    assert item["type_ids"][L - 1].item() == Vocabulary.EOS_ID
    assert item["type_ids"][L:].eq(Vocabulary.PAD_ID).all().item()


@pytest.mark.usefixtures("mocked_renderer")
def test_synthetic_dataset_resnet() -> None:
    vb = build_default_vocabs()
    spec = get_encoder_spec("resnet")
    ds = SyntheticOMRDataset(
        generator=MelodyGenerator(),
        renderer=StaffRenderer(),
        vocabs=vb,
        encoder_spec=spec,
        transform=spec.build_eval_transform(),
        length=2,
        seed=0,
        max_seq_len=128,
    )
    item = ds[0]
    assert item["pixel_values"].shape[0] == 1
    assert item["pixel_values"].shape[1] == 128


# ---------------------------------------------------------------------------
# Prerender + PreRenderedOMRDataset
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mocked_renderer")
def test_prerender_writes_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "val"
    manifest = prerender_split(
        out_dir,
        generator=MelodyGenerator(),
        renderer=StaffRenderer(),
        n=3,
        seed=1000,
    )
    assert manifest.exists()
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["image"] == f"{i:06d}.png"
        assert (out_dir / rec["image"]).exists()
        for k in ("type", "pitch", "rhythm", "attribute"):
            assert isinstance(rec[k], list)


@pytest.mark.usefixtures("mocked_renderer")
def test_prerendered_dataset_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "val"
    manifest = prerender_split(
        out_dir,
        generator=MelodyGenerator(),
        renderer=StaffRenderer(),
        n=2,
        seed=1000,
    )
    vb = build_default_vocabs()
    spec = get_encoder_spec("vit")
    ds = PreRenderedOMRDataset(
        manifest, vb, spec.build_eval_transform(), max_seq_len=128
    )
    assert len(ds) == 2
    item = ds[0]
    assert item["pixel_values"].shape == (3, 224, 224)
    assert item["type_ids"].shape == (128,)


# ---------------------------------------------------------------------------
# create_dataloaders smoke
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mocked_renderer")
def test_create_dataloaders_vit_smoke(tmp_path: Path) -> None:
    loaders = create_dataloaders(
        out_dir=tmp_path,
        encoder="vit",
        train_size=4,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        max_seq_len=128,
    )
    assert set(loaders) == {"train", "val", "test"}
    batch = next(iter(loaders["train"]))
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    for key in ("type_ids", "pitch_ids", "rhythm_ids", "attribute_ids"):
        assert batch[key].shape[0] == 2
    assert (tmp_path / "vocab" / "type.json").exists()


@pytest.mark.usefixtures("mocked_renderer")
def test_create_dataloaders_resnet_smoke(tmp_path: Path) -> None:
    loaders = create_dataloaders(
        out_dir=tmp_path,
        encoder="resnet",
        train_size=4,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        max_seq_len=128,
    )
    batch = next(iter(loaders["train"]))
    assert batch["pixel_values"].dim() == 4
    assert batch["pixel_values"].shape[1] == 1  # grayscale
    assert batch["pixel_values"].shape[2] == 128
