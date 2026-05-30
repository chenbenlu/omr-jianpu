from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartDecoder, shift_tokens_right

from src.data import Vocabulary, VocabBundle
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
