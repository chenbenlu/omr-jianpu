from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LossWeights:
    type: float = 1.0
    pitch: float = 1.0
    rhythm: float = 1.0
    attribute: float = 1.0


@dataclass
class MaskNullInLoss:
    pitch: bool = False
    rhythm: bool = False
    attribute: bool = False


@dataclass
class ModelConfig:
    d_model: int = 384
    decoder_layers: int = 4  # 在 CTC 模式下，這會直接複用為 RNN (LSTM) 的層數
    decoder_heads: int = 6
    decoder_ffn_dim: int = 1536
    dropout: float = 0.1
    max_decoder_positions: int = 512
    scale_embedding: bool = True
    # EOS is ~1 token per ~50 in the type stream, so its gradient is swamped by
    # the frequent note/rest classes and the model under-learns when to stop.
    # >1.0 up-weights EOS in the type-head cross-entropy to counter this.
    eos_weight: float = 1.0
    loss_weights: LossWeights = field(default_factory=LossWeights)
    mask_null_in_loss: MaskNullInLoss = field(default_factory=MaskNullInLoss)

    # === feature/B-crnn-and-ctr 新增：CRNN + CTC 模式專用參數 ===
    # 預設為 False，確保舊的 vit.yaml 和 resnet.yaml 不需要修改也能正常運作
    use_ctc: bool = False

    # BiLSTM 的隱藏層維度，預設 256（雙向展開後特徵維度會變成 256 * 2 = 512）
    rnn_hidden_dim: int = 256

    # 是否啟用雙向 RNN，光學字元/樂譜識別（OMR）通常設定為 True
    rnn_bidirectional: bool = True
