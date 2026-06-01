from __future__ import annotations

from typing import Sequence

import torch

from src.postproc.decode import TokenTuple


def ctc_greedy_decode(
    preds: Sequence[int] | torch.Tensor,
    id_to_joint_token: dict[int, tuple[str, str | None, str | None, str | None]],
    blank_id: int = 0,
) -> list[TokenTuple]:
    """Performs CTC Greedy Decoding on a single token ID sequence.

    Collapses consecutive duplicate tokens, removes the CTC <BLANK> tokens,
    and unpacks the remaining joint IDs back into standard 4-tuple TokenTuples.

    Args:
        preds: A sequence or 1D Tensor of raw class IDs predicted by the CRNN model.
        id_to_joint_token: Mapping from a single unified CTC integer ID back to
                           the original 4-tuple string token (type, pitch, rhythm, attribute).
        blank_id: The index corresponding to the <BLANK> token in the CTC vocab (default: 0).

    Returns:
        A list of TokenTuple items, perfectly aligned with Member C's state-machine
        and metrics contract.
    """
    # 1. 如果輸入是 PyTorch Tensor，將其轉換為標準 Python List 處理
    if isinstance(preds, torch.Tensor):
        pred_ids = preds.detach().cpu().tolist()
    else:
        pred_ids = list(preds)

    # 2. 步驟一：去重 (Collapse consecutive duplicates)
    collapsed_ids: list[int] = []
    previous_id = -1
    for token_id in pred_ids:
        if token_id != previous_id:
            collapsed_ids.append(token_id)
            previous_id = token_id

    # 3. 步驟二：移除 Blank 標籤 (Filter out CTC Blank tokens)
    # 同時過濾掉潛在的 0 (PAD)，確保序列純淨
    final_ids = [tid for tid in collapsed_ids if tid != blank_id]

    # 4. 步驟三：反向拆解還原為四分頭的 TokenTuple
    decoded_tuples: list[TokenTuple] = []
    for tid in final_ids:
        if tid in id_to_joint_token:
            # 找到對應的複合鍵，直接加入結果
            decoded_tuples.append(id_to_joint_token[tid])
        else:
            # 防禦性工程：若遇到未知的 ID（或模型噴出異常類別），退回系統標準 UNK
            # 這能確保下游簡譜渲染器渲染出 '?' 而不會發生崩潰
            decoded_tuples.append(("<UNK>", None, None, None))

    return decoded_tuples


def ctc_greedy_decode_batch(
    batch_preds: torch.Tensor,
    batch_lengths: torch.Tensor,
    id_to_joint_token: dict[int, tuple[str, str | None, str | None, str | None]],
    blank_id: int = 0,
) -> list[list[TokenTuple]]:
    """Batched version of CTC greedy decoding for validation loops and batch inference.

    Automatically honors the dynamic sequence length of each sample to avoid
    decoding trailing padding tokens.

    Args:
        batch_preds: Tensor of shape (B, T) containing predicted token IDs.
        batch_lengths: Tensor of shape (B,) containing the valid feature length for each row
                       (emitted by OMRCRNNModel.predict_greedy).
        id_to_joint_token: Mapping from a single unified CTC integer ID back to 4-tuple string token.
        blank_id: The index corresponding to the <BLANK> token in the CTC vocab (default: 0).

    Returns:
        A list of lists, where each sublist contains TokenTuple items for that batch sample.
    """
    B = batch_preds.size(0)
    batch_results: list[list[TokenTuple]] = []

    for i in range(B):
        # 根據模型傳過來的有效長度遮罩，只截取有效的時間步長 (扣除右側補白區域)
        valid_length = int(batch_lengths[i].item())
        sample_preds = batch_preds[i, :valid_length]

        # 呼叫單一序列解碼
        decoded_sample = ctc_greedy_decode(
            preds=sample_preds,
            id_to_joint_token=id_to_joint_token,
            blank_id=blank_id,
        )
        batch_results.append(decoded_sample)

    return batch_results
