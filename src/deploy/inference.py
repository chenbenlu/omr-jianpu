"""Reusable inference API: staff image -> Jianpu text.

Loads a trained VED checkpoint once and runs the full pipeline
(image -> encoder transform -> 4-head decoder -> postproc) on demand. Used by
both the CLI (`scripts/predict_jianpu.py`) and the Streamlit demo
(`src/deploy/app.py`).

The encoder type is not stored inside the checkpoint, so it is inferred from
the checkpoint directory name (`vit-...`, `resnet-...`) unless given explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from src.data import build_default_vocabs, get_encoder_spec
from src.model import ModelConfig, OMRModel
from src.model.encoders import build_encoder
from src.postproc import (
    JianpuRenderConfig,
    TokenTuple,
    ids_to_jianpu,
    ids_to_tuples,
)

ImageInput = Image.Image | np.ndarray | str | Path

# Architecture of the released checkpoints (configs/model/{vit,resnet}.yaml).
# Must match training exactly or load_state_dict produces garbage.
DEFAULT_MODEL_CONFIG = ModelConfig(
    d_model=384,
    decoder_layers=4,
    decoder_heads=6,
    decoder_ffn_dim=1536,
    dropout=0.1,
    max_decoder_positions=512,
)

_STEP_BEST_RE = re.compile(r"step-(\d+)-best")


@dataclass
class JianpuPrediction:
    jianpu: str
    tuples: list[TokenTuple]
    type_ids: list[int]
    pitch_ids: list[int]
    rhythm_ids: list[int]
    attribute_ids: list[int]
    length: int


def _infer_encoder_name(ckpt_dir: Path) -> str:
    """Parse the leading encoder token from a `<encoder>-<timestamp>` dir name.

    Walks up parents so a `step-N-best` subdir also resolves.
    """
    for part in (ckpt_dir.name, *(p.name for p in ckpt_dir.parents)):
        head = part.split("-", 1)[0]
        if head in {"vit", "resnet", "crnn"}:
            return head
    raise ValueError(
        f"cannot infer encoder from {ckpt_dir!r}; pass encoder= explicitly"
    )


def _resolve_checkpoint(ckpt_dir: Path) -> Path:
    """Return the directory holding `model.safetensors`.

    Accepts either a leaf checkpoint dir or a run dir containing many
    `step-N-best/` subdirs; in the latter case picks the highest step (the
    best val-SER snapshot saved last).
    """
    if (ckpt_dir / "model.safetensors").exists():
        return ckpt_dir
    candidates = [
        (int(m.group(1)), p)
        for p in ckpt_dir.iterdir()
        if p.is_dir() and (m := _STEP_BEST_RE.fullmatch(p.name))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no model.safetensors and no step-N-best/ subdir under {ckpt_dir}"
        )
    return max(candidates, key=lambda c: c[0])[1]


class OMRInferencer:
    def __init__(
        self,
        ckpt_dir: str | Path,
        encoder: str | None = None,
        device: str | None = None,
        model_config: ModelConfig | None = None,
        max_length: int = 64,
    ) -> None:
        ckpt_dir = Path(ckpt_dir)
        self.encoder_name = encoder or _infer_encoder_name(ckpt_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.vocabs = build_default_vocabs()
        # 因為 CRNN 的視覺主幹是 ResNet，如果模型架構是 crnn，影像規格請自動對齊 resnet
        spec_name = "resnet" if self.encoder_name == "crnn" else self.encoder_name
        self.spec = get_encoder_spec(spec_name)
        self.transform = self.spec.build_eval_transform()
        cfg = model_config or DEFAULT_MODEL_CONFIG

        # Only ViT downloads pretrained weights; build the bare arch and let the
        # checkpoint supply the trained weights. ResNet has no such kwarg.
        build_kwargs = {"pretrained": False} if self.encoder_name == "vit" else {}
        encoder_mod = build_encoder(self.spec, cfg.d_model, **build_kwargs)

        weights_dir = _resolve_checkpoint(ckpt_dir)
        # ==========================================
        # ==== 新增程式碼：紀錄當前 Batch 內影像壓扁後的 W 軸時間軸長度
        # ==========================================
        # --- 新增：CRNN 條件分支 ---
        if self.encoder_name == "crnn":
            import json

            from src.model.crnn import OMRCRNNModel

            # 讓推論器自動去 Checkpoint 目錄下尋找訓練時存好的聯名詞表
            vocab_json_path = weights_dir / "ctc_vocab.json"

            if vocab_json_path.exists():
                with vocab_json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)

                # 因為 JSON 的 key 預設是字串，載入後需將 key 轉回 int，value 轉回 tuple
                self.id_to_joint_token = {
                    int(k): tuple(v) for k, v in payload["id_to_joint_token"].items()
                }
                ctc_vocab_size = len(self.id_to_joint_token)
            else:
                # 防禦性的一個後路：如果檔案不存在，給予一個預設值或拋出異常
                self.id_to_joint_token = {}
                ctc_vocab_size = 2500  # 與 configs/model/crnn.yaml 一致

            self.model = OMRCRNNModel(
                encoder=encoder_mod, ctc_vocab_size=ctc_vocab_size, d_model=cfg.d_model
            )
        else:
            self.model = OMRModel(encoder=encoder_mod, vocabs=self.vocabs, cfg=cfg)

        state = load_file(str(weights_dir / "model.safetensors"))
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        self.checkpoint_path = weights_dir

    def _to_pixel_values(self, image: ImageInput) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
        return self.transform(image=arr)["image"]

    @torch.no_grad()
    def predict_batch(self, images: list[ImageInput]) -> list[JianpuPrediction]:
        tensors = [self._to_pixel_values(im) for im in images]
        pixel_values = torch.stack(tensors).to(self.device)
        jcfg = JianpuRenderConfig(emit_header=True)
        # ==========================================
        # ==== 新增程式碼：攔截推論流，導入 CTC 貪婪解碼
        # ==========================================
        # --- 新增：CRNN 貪婪解碼分支 ---
        if self.encoder_name == "crnn":
            from src.postproc.ctc_decode import ctc_greedy_decode

            # 1. 執行 CRNN 非自迴歸前向預測，拿到全時間軸的 Argmax IDs 與遮罩有效長度
            preds, lengths = self.model.predict_greedy(pixel_values, blank_id=0)

            results: list[JianpuPrediction] = []
            for i in range(len(images)):
                valid_len = int(lengths[i].item())
                # 2. 呼叫新寫的後處理，進行去重與 Blank 過濾，直接還原成 4-Tuple
                tuples = ctc_greedy_decode(
                    preds[i, :valid_len], self.id_to_joint_token, blank_id=0
                )

                # 3. 反向編碼回獨立 IDs，確保 100% 完美貼合前端 UI 的 DataFrame 可視化組態
                t = self.vocabs.type.encode([x[0] for x in tuples])
                p = self.vocabs.pitch.encode([x[1] for x in tuples])
                r = self.vocabs.rhythm.encode([x[2] for x in tuples])
                a = self.vocabs.attribute.encode([x[3] for x in tuples])

                try:
                    from src.postproc import tuples_to_jianpu

                    jianpu = tuples_to_jianpu(tuples, jcfg)
                except Exception:
                    # A wild prediction (e.g. an out-of-range key signature from an
                    # untrained model or an off-distribution image) can make postproc
                    # raise. Keep predict() total — surface an empty render rather
                    # than crashing the caller / UI.
                    jianpu = ""

                results.append(
                    JianpuPrediction(
                        jianpu=jianpu,
                        tuples=tuples,
                        type_ids=t,
                        pitch_ids=p,
                        rhythm_ids=r,
                        attribute_ids=a,
                        length=len(tuples),
                    )
                )
            return results
        # --- 原有 Transformer Decoder 分支 ---
        gen = self.model.generate(pixel_values, max_length=self.max_length)

        results: list[JianpuPrediction] = []
        for i in range(len(images)):
            length = int(gen.lengths[i].item())
            t = gen.type_ids[i][:length].tolist()
            p = gen.pitch_ids[i][:length].tolist()
            r = gen.rhythm_ids[i][:length].tolist()
            a = gen.attribute_ids[i][:length].tolist()
            try:
                jianpu = ids_to_jianpu(t, p, r, a, self.vocabs, jcfg)
            except Exception:
                # A wild prediction (e.g. an out-of-range key signature from an
                # untrained model or an off-distribution image) can make postproc
                # raise. Keep predict() total — surface an empty render rather
                # than crashing the caller / UI.
                jianpu = ""
            results.append(
                JianpuPrediction(
                    jianpu=jianpu,
                    tuples=ids_to_tuples(t, p, r, a, self.vocabs, strict=False),
                    type_ids=t,
                    pitch_ids=p,
                    rhythm_ids=r,
                    attribute_ids=a,
                    length=length,
                )
            )
        return results

    def predict(self, image: ImageInput) -> JianpuPrediction:
        return self.predict_batch([image])[0]
