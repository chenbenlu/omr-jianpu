from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from src.data import EncoderSpec


@dataclass
class EncoderOutput:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None


class ViTEncoder(nn.Module):
    DEFAULT_PRETRAINED = "google/vit-base-patch16-224-in21k"

    def __init__(
        self,
        d_model: int,
        pretrained: bool = True,
        vit_config_kwargs: dict[str, Any] | None = None,
        encoder_spec: EncoderSpec | None = None,
    ) -> None:
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained(
                self.DEFAULT_PRETRAINED, add_pooling_layer=False
            )
        else:
            cfg = ViTConfig(**(vit_config_kwargs or {}))
            self.vit = ViTModel(cfg, add_pooling_layer=False)
        hidden = self.vit.config.hidden_size
        self.proj = nn.Identity() if hidden == d_model else nn.Linear(hidden, d_model)

    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput:
        out = self.vit(pixel_values=pixel_values)
        # Drop [CLS] before cross-attention to avoid attention-weight collapse.
        hidden = out.last_hidden_state[:, 1:, :]
        hidden = self.proj(hidden)
        return EncoderOutput(hidden_states=hidden, attention_mask=None)


class ResNetEncoder(nn.Module):
    _STRIDES: tuple[tuple[int, int], ...] = (
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 1),
        (2, 1),
    )
    _CHANNELS: tuple[int, ...] = (32, 64, 128, 256, 256)

    def __init__(
        self,
        d_model: int,
        encoder_spec: EncoderSpec | None = None,
    ) -> None:
        super().__init__()
        in_channels = 1 if encoder_spec is None else encoder_spec.channels
        blocks: list[nn.Module] = []
        c_in = in_channels

        # === feature/B-crnn-and-ctr 修改：動態計算經過 CNN 卷積層後最終保留的特徵圖高度 H ===
        h_out = (
            encoder_spec.target_height
            if (encoder_spec and encoder_spec.target_height)
            else 128
        )
        for stride in self._STRIDES:
            # 根據 kernel_size=3, padding=1, stride=s 計算特徵圖尺寸變化
            h_out = (h_out + 2 * 1 - 3) // stride[0] + 1

        for c_out, stride in zip(self._CHANNELS, self._STRIDES):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        c_in,
                        c_out,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )
            )
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        # === feature/B-crnn-and-ctr 修改：將 Linear 層的輸入特徵維度改為 最終通道數 * 最終高度 ===
        # 原本是：self.proj = nn.Linear(self._CHANNELS[-1], d_model)
        self.proj = nn.Linear(self._CHANNELS[-1] * h_out, d_model)

    def _pad_mask(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # A column is "padding-or-empty" iff every channel-pixel equals +1.0
        # (normalized white). Propagate the mask through the same kernel/stride
        # sequence as the CNN so the mask shape matches the feature map exactly.
        raw = (pixel_values != 1.0).any(dim=1, keepdim=True).float()
        for stride in self._STRIDES:
            raw = F.avg_pool2d(raw, kernel_size=3, stride=stride, padding=1)
        col_mask = (raw.mean(dim=2) > 0.5).squeeze(1)
        return col_mask.to(torch.long)

    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput:
        feat = pixel_values
        for block in self.blocks:
            feat = block(feat)
        # === feature/B-crnn-and-ctr 修改：重寫壓扁與維度轉換邏輯（Pitch Fix） ===
        # 原本的程式碼會直接將高度壓扁：feat.mean(dim=2) -> 所以才會遺失所有垂直高度（音高）特徵
        # 改裝後：feat 形狀為 (B, C, H, W)
        B, C, H, W = feat.shape

        # 1. 調整維度順序，將寬度 W 移到時間序列軸 (B, W, C, H)
        seq = feat.permute(0, 3, 1, 2)
        # 2. 將同一個時間點（同一個 W）的通道特徵與高度空間特徵展平合併為 (B, W, C * H)
        seq = seq.reshape(B, W, C * H)
        # 3. 映射回標準的 d_model 維度，輸出為 (B, W, d_model)
        seq = self.proj(seq)
        mask = self._pad_mask(pixel_values)
        return EncoderOutput(hidden_states=seq, attention_mask=mask)


def build_encoder(
    spec: EncoderSpec,
    d_model: int,
    **kwargs: Any,
) -> nn.Module:
    if spec.name == "vit":
        return ViTEncoder(d_model=d_model, encoder_spec=spec, **kwargs)
    if spec.name == "resnet":
        return ResNetEncoder(d_model=d_model, encoder_spec=spec, **kwargs)
    raise KeyError(f"Unknown encoder spec name: {spec.name!r}")
