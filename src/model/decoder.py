from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartDecoder, shift_tokens_right

from src.data import VocabBundle, Vocabulary
from src.model.config import ModelConfig

_STREAMS: tuple[str, ...] = ("type", "pitch", "rhythm", "attribute")


class MultiHeadDecoder(nn.Module):
    """BartDecoder body with four input embeddings (summed) and four output heads.

    Training path (`forward`): receives raw label streams of shape (B, L). The
    data emits `[content..., EOS, PAD...]` with NO leading BOS, so we build the
    decoder input with `shift_tokens_right` (prepend `BOS`, drop the last
    position). The decoder then learns the `BOS -> first token` transition —
    which is exactly the state `generate()` starts from. Logits are full length
    L and align with the raw labels directly, so the loss does no slicing.

    Inference path (`step`): single-step forward fed by the caller's most
    recent 4-tuple of generated IDs; KV cache + position offset are handled by
    `BartDecoder` via the `past_key_values` / `cache_position` plumbing.
    """

    def __init__(self, vocabs: VocabBundle, cfg: ModelConfig) -> None:
        super().__init__()
        self.vocabs = vocabs
        self.d_model = cfg.d_model
        # embed_scale lives on the BartScaledWordEmbedding in HF, but we never
        # call BartDecoder.embed_tokens (we feed inputs_embeds), so derive it
        # ourselves and apply before passing into BartDecoder.
        self.embed_scale = math.sqrt(cfg.d_model) if cfg.scale_embedding else 1.0

        bart_cfg = BartConfig(
            d_model=cfg.d_model,
            decoder_layers=cfg.decoder_layers,
            decoder_attention_heads=cfg.decoder_heads,
            decoder_ffn_dim=cfg.decoder_ffn_dim,
            max_position_embeddings=cfg.max_decoder_positions,
            dropout=cfg.dropout,
            scale_embedding=cfg.scale_embedding,
            pad_token_id=Vocabulary.PAD_ID,
            bos_token_id=Vocabulary.BOS_ID,
            eos_token_id=Vocabulary.EOS_ID,
            decoder_start_token_id=Vocabulary.BOS_ID,
            # The BartDecoder owns an unused embed_tokens — we always feed
            # inputs_embeds. Make the table tiny so it doesn't waste memory.
            vocab_size=8,
            use_cache=True,
        )
        self.bart = BartDecoder(bart_cfg)

        # nn.ModuleDict forbids keys that collide with attributes on Module
        # (e.g., `type`), so prefix them. Public-facing names stay clean via
        # the _STREAMS tuple and the head_for/embed_for helpers below.
        self.embeddings = nn.ModuleDict(
            {
                f"emb_{name}": nn.Embedding(
                    len(vocab), cfg.d_model, padding_idx=Vocabulary.PAD_ID
                )
                for name, vocab in vocabs
            }
        )
        self.heads = nn.ModuleDict(
            {
                f"head_{name}": nn.Linear(cfg.d_model, len(vocab))
                for name, vocab in vocabs
            }
        )

    def _embed(self, ids: dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.embeddings[f"emb_{_STREAMS[0]}"](ids[_STREAMS[0]])
        for name in _STREAMS[1:]:
            emb = emb + self.embeddings[f"emb_{name}"](ids[name])
        return emb * self.embed_scale

    def forward(
        self,
        ids: dict[str, torch.Tensor],
        decoder_attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        # Build decoder inputs by prepending BOS (shift_tokens_right). Labels
        # have no leading BOS, so inputs = [BOS, c0, ..., c_{L-2}] and the raw
        # labels are the targets. Logits are length L, aligned with labels.
        in_ids = {
            n: shift_tokens_right(ids[n], Vocabulary.PAD_ID, Vocabulary.BOS_ID)
            for n in _STREAMS
        }
        # Shift the padding mask the same way: BOS at position 0 is always
        # valid, the rest tracks the labels shifted right by one.
        dec_mask = None
        if decoder_attention_mask is not None:
            ones = torch.ones_like(decoder_attention_mask[:, :1])
            dec_mask = torch.cat([ones, decoder_attention_mask[:, :-1]], dim=1)

        inputs_embeds = self._embed(in_ids)
        out = self.bart(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        logits = {name: self.heads[f"head_{name}"](hidden) for name in _STREAMS}
        return {"logits": logits}

    def step(
        self,
        ids: dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        past_key_values: Any,
    ) -> tuple[dict[str, torch.Tensor], Any]:
        # Single decoder step over (B, 1) IDs per stream. Caller manages the
        # cache: past_key_values=None on step 0, returned cache on subsequent
        # steps. BartDecoder uses past_key_values.get_seq_length() to derive
        # the position offset.
        inputs_embeds = self._embed(ids)
        out = self.bart(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        logits = {name: self.heads[f"head_{name}"](hidden) for name in _STREAMS}
        return logits, out.past_key_values


# === feature/B-crnn-and-ctr 新增：CRNN + CTC 非自迴歸時序解碼器 ===
class MultiHeadCTCDecoder(nn.Module):
    """基於 BiLSTM + 多頭獨立投影的 CTC 解碼器。

    接收形狀為 (B, W, d_model) 的影像序列特徵，透過深層雙向 LSTM 進行上下文時序建模，
    最終輸出 4 個分頭的預測 Logits。各分頭的類別數自動擴充 1 位以容納 CTC Blank Token。
    """

    def __init__(self, vocabs: VocabBundle, cfg: ModelConfig) -> None:
        super().__init__()
        self.vocabs = vocabs

        # 1. 建立循環神經網路（複用原有的 decoder_layers 作為 LSTM 堆疊層數）
        self.rnn = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.rnn_hidden_dim,
            num_layers=cfg.decoder_layers,
            batch_first=True,
            bidirectional=cfg.rnn_bidirectional,
            dropout=cfg.dropout if cfg.decoder_layers > 1 else 0.0,
        )

        # 計算 RNN 雙向展開後的最終特徵維度
        rnn_out_dim = cfg.rnn_hidden_dim * (2 if cfg.rnn_bidirectional else 1)

        # 2. 建立 4 個並行的線性預測頭
        # Key point：out_features 設為 len(vocab) + 1，多出來的最後一個位置保留給 CTC Blank
        self.heads = nn.ModuleDict(
            {
                f"head_{name}": nn.Linear(rnn_out_dim, len(vocab) + 1)
                for name, vocab in vocabs
            }
        )

        # 3. 記錄每個分頭對應的 Blank ID (即原 Vocabulary 長度值)
        self.blank_ids = {name: len(vocab) for name, vocab in vocabs}

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """CTC 前向傳播計算。

        Args:
            encoder_hidden_states: 影像編碼器輸出的時序特徵，形狀為 (B, W, d_model)
            encoder_attention_mask: 影像寬度填充遮罩（選填）

        Returns:
            dict: 包含 4 個標籤流 Logits Tensor 的字典，每個頭的形狀為 (B, W, len(vocab) + 1)
        """
        # 影像特徵通過雙向 LSTM 進行時序建模
        # rnn_out 的形狀: (B, W, rnn_out_dim)
        rnn_out, _ = self.rnn(encoder_hidden_states)

        # 分別投影到 4 個並行的獨立詞彙表分類空間
        logits = {name: self.heads[f"head_{name}"](rnn_out) for name, _ in self.vocabs}

        return logits
