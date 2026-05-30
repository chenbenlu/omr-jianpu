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
        if head in {"vit", "resnet"}:
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
        self.spec = get_encoder_spec(self.encoder_name)
        self.transform = self.spec.build_eval_transform()
        cfg = model_config or DEFAULT_MODEL_CONFIG

        # Only ViT downloads pretrained weights; build the bare arch and let the
        # checkpoint supply the trained weights. ResNet has no such kwarg.
        build_kwargs = {"pretrained": False} if self.encoder_name == "vit" else {}
        encoder_mod = build_encoder(self.spec, cfg.d_model, **build_kwargs)
        self.model = OMRModel(encoder=encoder_mod, vocabs=self.vocabs, cfg=cfg)
        weights_dir = _resolve_checkpoint(ckpt_dir)
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
        gen = self.model.generate(pixel_values, max_length=self.max_length)
        jcfg = JianpuRenderConfig(emit_header=True)

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
