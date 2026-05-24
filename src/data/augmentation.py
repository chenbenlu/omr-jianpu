from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

_DEFAULT_MEAN = (0.5, 0.5, 0.5)
_DEFAULT_STD = (0.5, 0.5, 0.5)


def _resize_and_pad(height: int, width: int) -> list[A.BasicTransform]:
    return [
        A.LongestMaxSize(max_size_hw=(height, width)),
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            position="top_left",
        ),
    ]


def _normalize_and_tensor(
    mean: tuple[float, float, float], std: tuple[float, float, float]
) -> list[A.BasicTransform]:
    return [A.Normalize(mean=mean, std=std), ToTensorV2()]


def get_train_augmentation(
    height: int = 128,
    width: int = 1024,
    mean: tuple[float, float, float] = _DEFAULT_MEAN,
    std: tuple[float, float, float] = _DEFAULT_STD,
) -> A.Compose:
    return A.Compose(
        [
            *_resize_and_pad(height, width),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.12), mean_range=(0.0, 0.0), p=0.3),
            A.Perspective(
                scale=(0.02, 0.05),
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                p=0.3,
            ),
            A.Rotate(
                limit=2,
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                p=0.4,
            ),
            *_normalize_and_tensor(mean, std),
        ]
    )


def get_eval_augmentation(
    height: int = 128,
    width: int = 1024,
    mean: tuple[float, float, float] = _DEFAULT_MEAN,
    std: tuple[float, float, float] = _DEFAULT_STD,
) -> A.Compose:
    return A.Compose(
        [*_resize_and_pad(height, width), *_normalize_and_tensor(mean, std)]
    )
