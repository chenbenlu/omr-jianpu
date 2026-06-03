from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.data import Vocabulary, VocabBundle
from src.model.config import ModelConfig
from src.model.decoder import MultiHeadDecoder

_STREAMS: tuple[str, ...] = ("type", "pitch", "rhythm", "attribute")


@dataclass
class GenerationOutput:
    type_ids: torch.LongTensor
    pitch_ids: torch.LongTensor
    rhythm_ids: torch.LongTensor
    attribute_ids: torch.LongTensor
    lengths: torch.LongTensor


class OMRModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        vocabs: VocabBundle,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = MultiHeadDecoder(vocabs, cfg)
        self.vocabs = vocabs
        self.cfg = cfg

    def forward(
        self,
        pixel_values: torch.Tensor,
        type_ids: torch.Tensor,
        pitch_ids: torch.Tensor,
        rhythm_ids: torch.Tensor,
        attribute_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> dict:
        enc = self.encoder(pixel_values)
        out = self.decoder(
            ids={
                "type": type_ids,
                "pitch": pitch_ids,
                "rhythm": rhythm_ids,
                "attribute": attribute_ids,
            },
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=enc.hidden_states,
            encoder_attention_mask=enc.attention_mask,
        )
        return {
            "logits": out["logits"],
            "encoder_hidden_states": enc.hidden_states,
            "encoder_attention_mask": enc.attention_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 512,
    ) -> GenerationOutput:
        device = pixel_values.device
        B = pixel_values.shape[0]
        enc = self.encoder(pixel_values)

        bos = torch.full((B, 1), Vocabulary.BOS_ID, dtype=torch.long, device=device)
        pad = Vocabulary.PAD_ID
        eos = Vocabulary.EOS_ID

        outs: dict[str, list[torch.Tensor]] = {n: [bos] for n in _STREAMS}
        # `finished[i]` flips True once row i's type head has emitted EOS.
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.full((B,), max_length, dtype=torch.long, device=device)

        past = None
        current = {n: bos for n in _STREAMS}

        for step_idx in range(1, max_length):
            logits, past = self.decoder.step(
                ids=current,
                encoder_hidden_states=enc.hidden_states,
                encoder_attention_mask=enc.attention_mask,
                past_key_values=past,
            )
            next_ids = {n: logits[n].argmax(dim=-1) for n in _STREAMS}  # each (B, 1)

            # For rows already finished, force PAD on every stream so the
            # downstream IdSeqs are clean.
            for n in _STREAMS:
                next_ids[n] = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_ids[n], pad),
                    next_ids[n],
                )

            # Record per-row stop index the first time type predicts EOS.
            type_emit_eos = next_ids["type"].squeeze(-1) == eos
            newly_finished = type_emit_eos & ~finished
            lengths = torch.where(
                newly_finished, torch.full_like(lengths, step_idx + 1), lengths
            )
            finished = finished | type_emit_eos

            for n in _STREAMS:
                outs[n].append(next_ids[n])
            current = next_ids

            if bool(finished.all().item()):
                break

        return GenerationOutput(
            type_ids=torch.cat(outs["type"], dim=1),
            pitch_ids=torch.cat(outs["pitch"], dim=1),
            rhythm_ids=torch.cat(outs["rhythm"], dim=1),
            attribute_ids=torch.cat(outs["attribute"], dim=1),
            lengths=lengths,
        )
