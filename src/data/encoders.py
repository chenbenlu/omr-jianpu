from __future__ import annotations

from collections.abc import Callable
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

    def build_train_transform(self, aug_profile: str = "default") -> A.Compose:
        return _build_transform(self, train=True, aug_profile=aug_profile)

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


# --- Augmentation profiles ---------------------------------------------------
#
# A profile maps to two op groups around the resize/pad stage:
#   slot A: photometric ops, run BEFORE LongestMaxSize/PadIfNeeded so they act on
#           the native-resolution image (realistic JPEG/ISO/blur/noise) and never
#           paint the white padding margins.
#   slot B: geometric ops only, run AFTER padding on the (H, W) canvas; exposed
#           regions are filled white to match the pad + Normalize(mean=0.5).
# Profiles are consulted only for train transforms; eval gets neither slot.

AugOps = list[A.BasicTransform]


def _default_aug(spec: EncoderSpec) -> tuple[AugOps, AugOps]:
    # Legacy photometric ops, historically applied AFTER pad. Kept in slot B so
    # the validated baseline recipe stays byte-identical.
    slot_b: AugOps = [
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.5), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=0.3),
    ]
    return [], slot_b


# `photo` is the union of three photometric/geometric component groups so the
# ablation profiles below can drop one group at a time (leave-one-out). Geometric
# warps are kept mild because pitch is absolute vertical position.


def _lighting_ops() -> AugOps:
    # Uneven illumination of a printed page under a phone camera.
    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RandomShadow(num_shadows_limit=(1, 2), p=0.25),
    ]


def _degrade_ops(spec: EncoderSpec) -> AugOps:
    # Optical + sensor degradation: blur, grain, JPEG. Photometric (pre-resize).
    ops: AugOps = [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.2, 2.0)),
                A.Defocus(radius=(1, 3)),
            ],
            p=0.4,
        ),
        A.GaussNoise(std_range=(0.02, 0.10), mean_range=(0.0, 0.0), p=0.3),
    ]
    if spec.channels == 3:
        # ISO grain and JPEG artifacts are defined on 3-channel camera images.
        ops.append(A.ISONoise(p=0.2))
        ops.append(A.ImageCompression(quality_range=(40, 85), p=0.4))
    return ops


def _geometric_ops() -> AugOps:
    # Camera-angle skew: applied AFTER pad (slot B), white fill for exposed areas.
    return [
        A.OneOf(
            [
                A.Affine(
                    scale=(0.92, 1.08),
                    rotate=(-4, 4),
                    shear=(-3, 3),
                    translate_percent=(0.0, 0.03),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=1.0,
                ),
                A.Perspective(
                    scale=(0.02, 0.05),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=1.0,
                ),
            ],
            p=0.6,
        ),
    ]


def _photo_aug(spec: EncoderSpec) -> tuple[AugOps, AugOps]:
    return _lighting_ops() + _degrade_ops(spec), _geometric_ops()


def _photo_no_geom(spec: EncoderSpec) -> tuple[AugOps, AugOps]:
    return _lighting_ops() + _degrade_ops(spec), []


def _photo_no_light(spec: EncoderSpec) -> tuple[AugOps, AugOps]:
    return _degrade_ops(spec), _geometric_ops()


def _photo_no_degrade(spec: EncoderSpec) -> tuple[AugOps, AugOps]:
    return _lighting_ops(), _geometric_ops()


_AUG_PROFILES: dict[str, Callable[[EncoderSpec], tuple[AugOps, AugOps]]] = {
    "default": _default_aug,
    "photo": _photo_aug,
    "photo_no_geom": _photo_no_geom,
    "photo_no_light": _photo_no_light,
    "photo_no_degrade": _photo_no_degrade,
}


def _build_transform(
    spec: EncoderSpec, *, train: bool, aug_profile: str = "default"
) -> A.Compose:
    pre_resize_ops: AugOps = []
    resize_pad_ops: AugOps = []

    if spec.is_dynamic_width():
        # Fixed height, aspect-preserving width (channel convert fused in).
        def _shape_lambda(image: np.ndarray, **kwargs: object) -> np.ndarray:
            img = _to_channels(image, spec.channels)
            return _resize_to_height_then_clip(img, spec.target_height, spec.max_width)

        resize_pad_ops.append(A.Lambda(image=_shape_lambda, p=1.0))
    else:
        # Fixed-size: settle channels, then aspect-preserving resize + white-pad.
        assert spec.target_width is not None

        def _channel_lambda(image: np.ndarray, **kwargs: object) -> np.ndarray:
            return _to_channels(image, spec.channels)

        pre_resize_ops.append(A.Lambda(image=_channel_lambda, p=1.0))
        resize_pad_ops.append(
            A.LongestMaxSize(max_size_hw=(spec.target_height, spec.target_width))
        )
        resize_pad_ops.append(
            A.PadIfNeeded(
                min_height=spec.target_height,
                min_width=spec.target_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                position="top_left",
            )
        )

    slot_a: AugOps = []
    slot_b: AugOps = []
    if train:
        try:
            make = _AUG_PROFILES[aug_profile]
        except KeyError as exc:
            known = ", ".join(sorted(_AUG_PROFILES))
            raise KeyError(
                f"Unknown aug_profile '{aug_profile}'. Known: {known}"
            ) from exc
        slot_a, slot_b = make(spec)

    post_ops = [
        A.Normalize(mean=spec.normalize_mean, std=spec.normalize_std),
        ToTensorV2(),
    ]

    return A.Compose(pre_resize_ops + slot_a + resize_pad_ops + slot_b + post_ops)


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
