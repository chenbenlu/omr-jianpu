from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import VocabBundle, Vocabulary


def compute_ctc_loss(
    logits: torch.Tensor,
    ctc_targets: torch.Tensor,
    attention_mask: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 0,
    zero_infinity: bool = True,
) -> torch.Tensor:
    """Compute the Connectionist Temporal Classification (CTC) Loss for CRNN OMR.

    Args:
        logits: Unnormalized log probabilities from OMRCRNNModel of shape (B, T, V).
        ctc_targets: Ground-truth joint target IDs of shape (B, max_target_len).
                     Should be padded with -1 or blank_id (handled by ignore_index/reduction).
        attention_mask: Encoder attention mask of shape (B, T) propagated from ResNet,
                          where 1 indicates valid visual features, 0 indicates padded white space.
        target_lengths: Actual lengths of each joint target sequence in the batch, shape (B...).
        blank_id: The index corresponding to the <BLANK> token in the CTC vocab (default: 0).
        zero_infinity: Whether to zero infinite losses. Highly recommended for OMR early training
                       to avoid crash when input visual frames < ground-truth target tokens.

    Returns:
        A scalar tensor representing the average CTC loss of the batch.
    """
    # 1. 將 CRNN 輸出的 Logits 轉換成 CTC 期望的 Log Probabilities
    # F.log_softmax 沿著 類別字典維度 (dim=-1) 計算
    log_probs = F.log_softmax(logits, dim=-1)  # 基於模型輸出，形狀為 (B, T, V)

    # 由於 nn.CTCLoss 嚴格要求時間軸在第一維，我們透過 transpose 將 (B, T, V) 轉置為 (T, B, V)
    log_probs = log_probs.transpose(0, 1)  # 轉換為 (T, B, V)

    # 2. 初始化 PyTorch 內建的 CTCLoss
    # 移除了不支援的 batch_first=True 參數
    ctc_loss_fn = nn.CTCLoss(
        blank=blank_id,
        reduction="mean",
        zero_infinity=zero_infinity,
    )

    # 3. 從視覺編碼器的 attention_mask 計算出每個 Batch 樣本實際的輸入時間序列長度 (Input Lengths)
    # attention_mask 形狀為 (B, T)，1 代表有效特徵，加總即為該圖片壓扁後的實際序列長度
    input_lengths = attention_mask.sum(dim=-1).to(torch.long)  # (B,)

    # 4. 確保 target_lengths 與應有的型態 (torch.long) 一致
    target_lengths = target_lengths.to(torch.long)

    # 5. 防禦性工程檢查：若在不開 JIT 的一般訓練模式下，發現視覺特徵太短無法對齊標籤，印出警告
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        len_mismatch = input_lengths < target_lengths
        if len_mismatch.any():
            bad_indices = torch.where(len_mismatch)[0].tolist()
            print(
                f"[CTC Warning] Input length is shorter than target length at batch indices {bad_indices}. "
                f"input_lengths: {input_lengths[len_mismatch].tolist()} vs "
                f"target_lengths: {target_lengths[len_mismatch].tolist()}. "
                f"Gradients for these samples will be zeroed out safely via zero_infinity=True."
            )

    # 6. 計算 CTC 損失
    loss = ctc_loss_fn(
        log_probs=log_probs,
        targets=ctc_targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )

    return loss


def pack_streams_to_ctc_targets(
    batch: dict[str, torch.Tensor],
    vocabs: VocabBundle,
    joint_token_to_id: dict[tuple[str, str | None, str | None, str | None], int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """On-the-fly utility to convert multi-head parallel streams into a single joint CTC target tensor.

    This bridges Member A's 4-stream lock-step data contract and Member B's CTC loss input requirements.
    It automatically filters out <PAD>, <BOS>, <EOS> to ensure a clean alignment sequence.

    Args:
        batch: The raw batch dictionary emitted by the DataLoader containing 'type_ids', 'pitch_ids', etc.
        vocabs: The original default VocabBundle for single-head decoding lookups.
        joint_token_to_id: A dictionary mapping from a 4-tuple token to a single unified CTC integer ID.

    Returns:
        ctc_targets: LongTensor of shape (B, max_packed_target_len) padded with 0.
        target_lengths: LongTensor of shape (B,) containing actual lengths of each packed row.
    """
    device = batch["type_ids"].device
    B = batch["type_ids"].shape[0]

    # 解碼還原成原始字串，排除特殊控制 Token，再打包成一體化的 CTC Token ID
    packed_batch_list: list[torch.Tensor] = []
    lengths_list: list[int] = []

    for i in range(B):
        # 讀取當前 Sample 的有效長度 (排除尾部 PAD)
        L = int(batch["label_lengths"][i].item())

        t_ids = batch["type_ids"][i, :L].tolist()
        p_ids = batch["pitch_ids"][i, :L].tolist()
        r_ids = batch["rhythm_ids"][i, :L].tolist()
        a_ids = batch["attribute_ids"][i, :L].tolist()

        # 透過既存的各頭 Vocab 還原成文字
        # skip_special_tokens=True 會自動幫我們剔除 BOS / EOS / PAD
        t_tokens = vocabs.type.decode(t_ids, skip_special_tokens=True)
        p_tokens = vocabs.pitch.decode(p_ids, skip_special_tokens=True)
        r_tokens = vocabs.rhythm.decode(r_ids, skip_special_tokens=True)
        a_tokens = vocabs.attribute.decode(a_ids, skip_special_tokens=True)

        # 確保 lock-step 長度一致
        n_elements = len(t_tokens)

        packed_ids = []
        for j in range(n_elements):
            # 組裝成 4-Tuple 複合鍵
            token_tuple = (t_tokens[j], p_tokens[j], r_tokens[j], a_tokens[j])
            # 對照一體化詞表，若完全沒看過則回退至 UNK (通常對應聯合詞表的 UNK_ID=3)
            # 預設 joint_token_to_id 應在外部建好
            ctc_id = joint_token_to_id.get(token_tuple, Vocabulary.UNK_ID)
            packed_ids.append(ctc_id)

        packed_batch_list.append(torch.tensor(packed_ids, dtype=torch.long))
        lengths_list.append(len(packed_ids))

    # 動態 Padding 組裝成一個穩定的二維 Tensor B x Max_Target_Len
    max_target_len = max(lengths_list) if lengths_list else 1
    # CTC targets 補零 (通常 0 在 CTC 中也作為 PAD_ID 或者是 BLANK_ID，
    # 只要符合 nn.CTCLoss 的長度宣告，後面超過 target_length 的區域都會被自動忽略)
    ctc_targets = torch.zeros((B, max_target_len), dtype=torch.long, device=device)
    for i, row in enumerate(packed_batch_list):
        if len(row) > 0:
            ctc_targets[i, : len(row)] = row

    target_lengths = torch.tensor(lengths_list, dtype=torch.long, device=device)

    return ctc_targets, target_lengths
