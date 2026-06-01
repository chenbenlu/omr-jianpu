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
        self.proj = nn.Linear(self._CHANNELS[-1], d_model)

    def _pad_mask(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # A column is "padding-or-empty" iff every channel-pixel equals +1.0
        # (normalized white). Propagate the mask through the same kernel/stride
        # sequence as the CNN so the mask shape matches the feature map exactly.
        raw = (pixel_values != 1.0).any(dim=1, keepdim=True).float()
        for stride in self._STRIDES:
            raw = F.avg_pool2d(raw, kernel_size=3, stride=stride, padding=1)
        # ==========================================
        # ==== 修改程式碼：修改閾值與維度
        # ==========================================
        # raw 的形狀為 (B, 1, H_feat, W_feat)
        # 將閾值從 > 0.5 改為 > 1e-5 (只要有任何音符或譜線痕跡都算有效內容)
        # 將維度從 4D 確實降維成 2D 的 (B, W_feat) 矩陣，以符合 ctc_losses 的 input_lengths 預期
        col_mask = (raw.sum(dim=2) > 1e-5).to(torch.long)  # (B, 1, W_feat)
        return col_mask.squeeze(1)  # 確保輸出為 (B, W_feat)

    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput:
        feat = pixel_values
        for block in self.blocks:
            feat = block(feat)
        # Collapse vertical axis, treat width as the time dimension.
        seq = feat.mean(dim=2).transpose(1, 2).contiguous()
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
