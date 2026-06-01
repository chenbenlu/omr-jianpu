from __future__ import annotations

import torch
import torch.nn as nn

from src.model.encoders import EncoderOutput, ResNetEncoder


class OMRCRNNModel(nn.Module):
    """CRNN (ResNet + Bi-LSTM) for Optical Music Recognition with CTC Loss.

    This architecture treats the OMR task as a sequence-to-sequence alignment
    problem, replacing the autoregressive Transformer Decoder with a Bidirectional
    LSTM. It processes the sequence features extracted from the vertical-collapsed
    ResNet feature map and projects them onto the joint CTC vocabulary space.
    """

    def __init__(
        self,
        encoder: nn.Module,
        ctc_vocab_size: int,
        d_model: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            encoder: The visual encoder instance (expected to be ResNetEncoder).
            ctc_vocab_size: Total size of the joint CTC vocabulary (including <BLANK>).
            d_model: The feature dimension emitted by the encoder (default: 384).
            hidden_dim: Hidden dimension size for each directional LSTM layer.
            num_layers: Number of stacked LSTM layers (default: 2).
            dropout: Dropout probability applied between LSTM layers.
        """
        super().__init__()
        # 1. 視覺特徵提取器：直接複用專案既有的 ResNetEncoder
        if not isinstance(encoder, ResNetEncoder):
            raise TypeError(
                f"OMRCRNNModel expects a ResNetEncoder, got {type(encoder).__name__}. "
                "ViT cannot be used with CTC due to rigid grid spatial layout."
            )
        self.encoder = encoder
        self.d_model = d_model

        # 2. 序列時序建模層：雙向 LSTM (Bi-LSTM)
        # 由於是雙向，輸出的維度將會是 hidden_dim * 2
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. 投影層：將 RNN 特徵映射到一體化 CTC 詞表類別機率上
        self.fc = nn.Linear(hidden_dim * 2, ctc_vocab_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training and batched inference.

        Args:
            pixel_values: Input image tensor of shape (B, C, H, W).
                          For ResNet, expected shape is (B, 1, 128, W_dynamic).

        Returns:
            A dict containing:
                "logits": Tensor of shape (B, T, V) containing unnormalized log probabilities.
                "attention_mask": Tensor of shape (B, T) propagated from the encoder padding mask,
                                  where 1 indicates a real feature column and 0 indicates PAD.
        """
        # 1. 通過 ResNet 提取序列特徵：
        # hidden_states: (B, T, d_model), attention_mask: (B, T)
        enc_out: EncoderOutput = self.encoder(pixel_values)

        # 2. 通過 Bi-LSTM 時序融合層：
        # rnn_out: (B, T, hidden_dim * 2)
        rnn_out, _ = self.rnn(enc_out.hidden_states)

        # 3. 映射至詞表機率空間：
        # logits: (B, T, ctc_vocab_size)
        logits = self.fc(rnn_out)

        return {
            "logits": logits,
            "attention_mask": enc_out.attention_mask,
        }

    @torch.no_grad()
    def predict_greedy(
        self,
        pixel_values: torch.Tensor,
        blank_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding execution for a single or batched input images.

        Performs argmax over the vocabulary axis, exposes raw token IDs per time-step.
        The downstream `ctc_decode.py` will handle duplicate collapses and blank filtering.

        Args:
            pixel_values: Input image tensor of shape (B, C, H, W).
            blank_id: The index corresponding to the <BLANK> token in the CTC vocab.

        Returns:
            preds: LongTensor of shape (B, T) containing the argmax token IDs.
            lengths: LongTensor of shape (B,) containing the valid feature sequence length
                     for each row (derived from the encoder attention mask).
        """
        self.eval()
        out = self.forward(pixel_values)
        logits = out["logits"]  # (B, T, V)
        mask = out["attention_mask"]  # (B, T)

        # 取得每個時間點機率最大的 Token ID
        preds = logits.argmax(dim=-1)  # (B, T)

        # 根據 Encoder 傳過來的 mask 動態計算每個 Batch 的實際特徵長度
        # 因為 mask 中 1 代表有效，0 代表 PAD，加總即為該張圖片壓扁後的實際序列長度
        lengths = mask.sum(dim=-1)  # (B,)

        # 為了保持下游 IdSeqs 的乾淨度，強制將被 Mask 掉（補零）的區域全部刷成 blank_id
        preds = torch.where(mask == 1, preds, torch.full_like(preds, blank_id))

        return preds, lengths
