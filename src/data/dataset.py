from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.encoders import EncoderSpec
from src.data.generator import MelodyGenerator
from src.data.renderer import StaffRenderer
from src.data.vocabulary import VocabBundle, Vocabulary

_LABEL_KEYS = ("type", "pitch", "rhythm", "attribute")


@dataclass(frozen=True)
class _EncodedLabels:
    type_ids: list[int]
    pitch_ids: list[int]
    rhythm_ids: list[int]
    attribute_ids: list[int]
    length: int


def _encode_with_eos(
    raw: dict[str, list[str | None]], vocabs: VocabBundle, max_seq_len: int
) -> _EncodedLabels:
    n = len(raw["type"])
    if any(len(raw[k]) != n for k in _LABEL_KEYS):
        raise ValueError("label streams have unequal lengths")
    if n + 1 > max_seq_len:
        raise ValueError(
            f"sequence length {n}+1(EOS) exceeds max_seq_len={max_seq_len}"
        )

    type_ids = vocabs.type.encode(raw["type"]) + [Vocabulary.EOS_ID]
    pitch_ids = vocabs.pitch.encode(raw["pitch"]) + [Vocabulary.EOS_ID]
    rhythm_ids = vocabs.rhythm.encode(raw["rhythm"]) + [Vocabulary.EOS_ID]
    attribute_ids = vocabs.attribute.encode(raw["attribute"]) + [Vocabulary.EOS_ID]
    length = len(type_ids)

    if Vocabulary.UNK_ID in type_ids[:-1]:
        raise AssertionError("UNK in closed-set type stream — generator/vocab desync")
    if Vocabulary.UNK_ID in pitch_ids[:-1]:
        raise AssertionError("UNK in closed-set pitch stream — generator/vocab desync")
    if Vocabulary.UNK_ID in rhythm_ids[:-1]:
        raise AssertionError("UNK in closed-set rhythm stream — generator/vocab desync")
    if Vocabulary.UNK_ID in attribute_ids[:-1]:
        raise AssertionError(
            "UNK in closed-set attribute stream — generator/vocab desync"
        )

    pad = max_seq_len - length
    return _EncodedLabels(
        type_ids=type_ids + [Vocabulary.PAD_ID] * pad,
        pitch_ids=pitch_ids + [Vocabulary.PAD_ID] * pad,
        rhythm_ids=rhythm_ids + [Vocabulary.PAD_ID] * pad,
        attribute_ids=attribute_ids + [Vocabulary.PAD_ID] * pad,
        length=length,
    )


def _build_item(
    image_rgb: np.ndarray,
    raw_labels: dict[str, list[str | None]],
    transform: A.Compose,
    vocabs: VocabBundle,
    max_seq_len: int,
) -> dict[str, Any]:
    if image_rgb.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {image_rgb.dtype}")
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got {image_rgb.shape}")

    pixel_values: torch.Tensor = transform(image=image_rgb)["image"]
    enc = _encode_with_eos(raw_labels, vocabs, max_seq_len)
    return {
        "pixel_values": pixel_values,
        "type_ids": torch.tensor(enc.type_ids, dtype=torch.long),
        "pitch_ids": torch.tensor(enc.pitch_ids, dtype=torch.long),
        "rhythm_ids": torch.tensor(enc.rhythm_ids, dtype=torch.long),
        "attribute_ids": torch.tensor(enc.attribute_ids, dtype=torch.long),
        "label_length": enc.length,
    }


class SyntheticOMRDataset(Dataset):
    def __init__(
        self,
        generator: MelodyGenerator,
        renderer: StaffRenderer,
        vocabs: VocabBundle,
        encoder_spec: EncoderSpec,
        transform: A.Compose,
        length: int,
        seed: int,
        max_seq_len: int = 512,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be > 0")
        self.generator = generator
        self.renderer = renderer
        self.vocabs = vocabs
        self.encoder_spec = encoder_spec
        self.transform = transform
        self.length = int(length)
        self.seed = int(seed)
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        sample = self.generator.generate(self.seed, idx)
        image = self.renderer.render(sample.stream)
        return _build_item(
            image, sample.labels, self.transform, self.vocabs, self.max_seq_len
        )


class PreRenderedOMRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        vocabs: VocabBundle,
        transform: A.Compose,
        max_seq_len: int = 512,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.vocabs = vocabs
        self.transform = transform
        self.max_seq_len = int(max_seq_len)
        self._records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        if not self._records:
            raise ValueError(f"empty manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self._records[idx]
        img_path = self.root / rec["image"]
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        raw_labels = {k: rec[k] for k in _LABEL_KEYS}
        return _build_item(
            image, raw_labels, self.transform, self.vocabs, self.max_seq_len
        )
