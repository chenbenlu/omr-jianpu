from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.data import VocabBundle, Vocabulary
from src.model.config import ModelConfig
from src.model.decoder import MultiHeadCTCDecoder, MultiHeadDecoder

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
        # === feature/B-crnn-and-ctr 修改：根據 use_ctc 開關動態實例化對應的解碼器 ===
        if self.cfg.use_ctc:
            self.decoder = MultiHeadCTCDecoder(vocabs, cfg)
        else:
            self.decoder = MultiHeadDecoder(vocabs, cfg)

    def forward(
        self,
        pixel_values: torch.Tensor,
        type_ids: torch.Tensor | None = None,
        pitch_ids: torch.Tensor | None = None,
        rhythm_ids: torch.Tensor | None = None,
        attribute_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> dict:
        enc = self.encoder(pixel_values)
        # === feature/B-crnn-and-ctr 修改： 根據模式進行解碼分流 ===
        if self.cfg.use_ctc:
            # CTC 模式：直接將編碼特徵餵入時序解碼器
            logits = self.decoder(enc.hidden_states, enc.attention_mask)
            return {
                "logits": logits,
                "encoder_hidden_states": enc.hidden_states,
                "encoder_attention_mask": enc.attention_mask,
            }
        else:
            # 舊有自迴歸模式（保持向下相容）
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
        """非自迴歸 CTC 貪婪解碼 (Greedy Decoding) 與自迴歸生成演算法。"""
        self.eval()
        # === feature/B-crnn-and-ctr 修改： 也是根據模式進行分流 ===
        # 如果非 CTC 模式，走原本的逐字自迴歸生成邏輯(將這裡本來的內容包成函數了)
        if not self.cfg.use_ctc:
            return self._generate_autoregressive(pixel_values, max_length)

        # == 新增 CTC 模式專屬解碼流水線 ==
        device = pixel_values.device
        B = pixel_values.shape[0]

        # 1. 執行 Forward 取得全時間步的 Logits 分佈
        fwd_out = self.forward(pixel_values)
        logits = fwd_out["logits"]

        # 用於儲存這一個 Batch 內所有樣本去重壓縮後的動態結果
        decoded_batch_streams: dict[str, list[list[int]]] = {n: [] for n in _STREAMS}
        final_sample_lengths = []

        # 2. 逐一對 Batch 內的 Sample 進行時序 Collapse Process
        for b in range(B):
            max_stream_len_for_this_sample = 0

            for name, vocab in self.vocabs:
                blank_id = self.decoder.blank_ids[name]

                # 取出第 b 個樣本在整個圖像寬度時間軸 W 上的最大機率 ID 序列
                # logits[name] 形狀為 (B, W, V+1) -> argmax 後為 (W,)
                raw_time_steps = logits[name][b].argmax(dim=-1).tolist()

                # 執行 CTC 核心去重演算法
                collapsed_sequence = []
                prev_id = None
                for current_id in raw_time_steps:
                    if current_id != prev_id:  # rule1：壓縮連續重複的預測
                        if current_id != blank_id:  # rule2：剔除 CTC Blank 預測
                            collapsed_sequence.append(current_id)
                        prev_id = current_id

                # 樂譜合約安全機制：若預測序列中包含了數據端自帶的 EOS_ID，直接在此處做提早截斷
                if Vocabulary.EOS_ID in collapsed_sequence:
                    eos_idx = collapsed_sequence.index(Vocabulary.EOS_ID)
                    collapsed_sequence = collapsed_sequence[: eos_idx + 1]

                decoded_batch_streams[name].append(collapsed_sequence)

                # 追蹤 4 個分頭中解碼出來最長的序列長度
                if len(collapsed_sequence) > max_stream_len_for_this_sample:
                    max_stream_len_for_this_sample = len(collapsed_sequence)

            # 外部 postproc 模組預期切片長度包含開頭的 BOS_ID，故真實可用長度為 序列長度 + 1
            final_sample_lengths.append(max_stream_len_for_this_sample + 1)

        # 3. 矩陣對齊與補齊 (Collation & Padding)
        # 計算此 Batch 中最長符號長度，作為 Tensor 的寬度規格
        batch_max_len = max(final_sample_lengths)
        batch_max_len = max(batch_max_len, 16)  # 設定最小邊界

        padded_output_tensors = {}
        for name in _STREAMS:
            padded_list = []
            for b in range(B):
                # 關鍵向後相容合約：序列起頭強制置入 BOS_ID
                full_seq = [Vocabulary.BOS_ID] + decoded_batch_streams[name][b]

                # 計算需要用 PAD_ID 補齊的空間
                pad_size = batch_max_len - len(full_seq)
                if pad_size > 0:
                    full_seq = full_seq + [Vocabulary.PAD_ID] * pad_size
                else:
                    full_seq = full_seq[:batch_max_len]  # 截斷安全邊界
                padded_list.append(full_seq)

            padded_output_tensors[name] = torch.tensor(
                padded_list, dtype=torch.long, device=device
            )

        return GenerationOutput(
            type_ids=padded_output_tensors["type"],
            pitch_ids=padded_output_tensors["pitch"],
            rhythm_ids=padded_output_tensors["rhythm"],
            attribute_ids=padded_output_tensors["attribute"],
            lengths=torch.tensor(final_sample_lengths, dtype=torch.long, device=device),
        )

    def _generate_autoregressive(
        self,
        pixel_values: torch.Tensor,
        max_length: int,
    ) -> GenerationOutput:
        """原本的 BART 自迴歸解碼生成函數（保持完全不變，供 Vit 路線與向上相容使用）"""
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
