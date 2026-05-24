from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augmentation import get_eval_augmentation
from src.data.vocabulary import Vocabulary


def _resolve_image_path(sample_dir: Path, sid: str, use_camera: bool) -> Path:
    # Real Camera-PrIMuS ships `{sid}_distorted.jpg`; the test fixtures use `{sid}.jpg`.
    # Fall through to the clean PNG when no camera variant exists.
    if use_camera:
        for ext in ("_distorted.jpg", ".jpg"):
            candidate = sample_dir / f"{sid}{ext}"
            if candidate.exists():
                return candidate
    return sample_dir / f"{sid}.png"


class PrIMuSDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        vocabulary: Vocabulary,
        sample_ids: list[str],
        transform: A.Compose | None = None,
        use_camera: bool = False,
        max_seq_len: int = 512,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.vocabulary = vocabulary
        self.sample_ids = sample_ids
        self.transform = transform if transform is not None else get_eval_augmentation()
        self.use_camera = use_camera
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        sid = self.sample_ids[idx]
        sample_dir = self.root_dir / sid

        img_path = _resolve_image_path(sample_dir, sid, self.use_camera)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for sample '{sid}': {img_path}")

        img = np.array(Image.open(img_path).convert("RGB"))
        pixel_values: torch.Tensor = self.transform(image=img)["image"]

        semantic_path = sample_dir / f"{sid}.semantic"
        text = semantic_path.read_text(encoding="utf-8").strip()
        tokens = [t.strip() for t in text.split("\t") if t.strip()]

        # Labels are `tokens + EOS + PAD…` — no BOS. HF VisionEncoderDecoderModel
        # prepends BOS via `shift_tokens_right` using decoder_start_token_id.
        # Truncation here is defensive; create_dataloaders filters overlong
        # samples upstream so this branch is unreachable from that path.
        tokens = tokens[: self.max_seq_len - 1]
        ids = self.vocabulary.encode(tokens) + [Vocabulary.EOS_ID]
        label_length = len(ids)

        ids += [Vocabulary.PAD_ID] * (self.max_seq_len - label_length)
        labels = torch.tensor(ids, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "label_length": label_length,
        }
