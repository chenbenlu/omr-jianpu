from __future__ import annotations

from dataclasses import dataclass

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    channels: int
    target_height: int
    target_width: int | None
    max_width: int
    normalize_mean: tuple[float, ...]
    normalize_std: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {self.channels}")
        if len(self.normalize_mean) != self.channels:
            raise ValueError("normalize_mean length must match channels")
        if len(self.normalize_std) != self.channels:
            raise ValueError("normalize_std length must match channels")
        if self.target_width is not None and self.max_width < self.target_width:
            raise ValueError("max_width < target_width")

    def is_dynamic_width(self) -> bool:
        return self.target_width is None

    def build_train_transform(self) -> A.Compose:
        return _build_transform(self, train=True)

    def build_eval_transform(self) -> A.Compose:
        return _build_transform(self, train=False)


def _to_channels(img: np.ndarray, channels: int) -> np.ndarray:
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == channels:
        return img
    if channels == 1 and img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray[..., None]
    if channels == 3 and img.shape[-1] == 1:
        return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2RGB)
    raise ValueError(
        f"cannot reshape image from {img.shape[-1]} to {channels} channels"
    )


def _resize_to_height_then_clip(
    img: np.ndarray, height: int, max_width: int
) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0:
        raise ValueError("image height is 0")
    new_w = max(1, int(round(w * height / h)))
    resized = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = resized[..., None]
    if resized.shape[1] > max_width:
        # Center-crop to max_width (only rare overflows; protects collator memory).
        start = (resized.shape[1] - max_width) // 2
        resized = resized[:, start : start + max_width, :]
    return resized


def _build_transform(spec: EncoderSpec, *, train: bool) -> A.Compose:
    pre_ops: list[A.BasicTransform] = []

    if spec.is_dynamic_width():
        # Fixed height, aspect-preserving width.
        def _shape_lambda(image: np.ndarray, **kwargs: object) -> np.ndarray:
            img = _to_channels(image, spec.channels)
            return _resize_to_height_then_clip(img, spec.target_height, spec.max_width)

        pre_ops.append(A.Lambda(image=_shape_lambda, p=1.0))
    else:
        # Fixed-size: aspect-preserving resize then white-pad to (H, W).
        assert spec.target_width is not None

        def _channel_lambda(image: np.ndarray, **kwargs: object) -> np.ndarray:
            return _to_channels(image, spec.channels)

        pre_ops.append(A.Lambda(image=_channel_lambda, p=1.0))
        pre_ops.append(
            A.LongestMaxSize(max_size_hw=(spec.target_height, spec.target_width))
        )
        pre_ops.append(
            A.PadIfNeeded(
                min_height=spec.target_height,
                min_width=spec.target_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                position="top_left",
            )
        )

    aug_ops: list[A.BasicTransform] = []
    if train:
        aug_ops = [
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.5), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=0.3),
        ]

    post_ops = [
        A.Normalize(mean=spec.normalize_mean, std=spec.normalize_std),
        ToTensorV2(),
    ]

    return A.Compose(pre_ops + aug_ops + post_ops)


VIT_SPEC = EncoderSpec(
    name="vit",
    channels=3,
    target_height=224,
    target_width=224,
    max_width=224,
    normalize_mean=(0.5, 0.5, 0.5),
    normalize_std=(0.5, 0.5, 0.5),
)

RESNET_SPEC = EncoderSpec(
    name="resnet",
    channels=1,
    target_height=128,
    target_width=None,
    max_width=1600,
    normalize_mean=(0.5,),
    normalize_std=(0.5,),
)

ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "vit": VIT_SPEC,
    "resnet": RESNET_SPEC,
}


def get_encoder_spec(encoder: str | EncoderSpec) -> EncoderSpec:
    if isinstance(encoder, EncoderSpec):
        return encoder
    try:
        return ENCODER_REGISTRY[encoder]
    except KeyError as exc:
        known = ", ".join(sorted(ENCODER_REGISTRY))
        raise KeyError(f"Unknown encoder '{encoder}'. Known: {known}") from exc
