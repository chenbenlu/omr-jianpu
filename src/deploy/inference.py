"""Reusable inference API: staff image -> Jianpu text.

Loads a trained VED checkpoint once and runs the full pipeline
(image -> encoder transform -> {AR multi-head OR CRNN+CTC} decoder -> postproc)
on demand. Used by both the CLI (`scripts/predict_jianpu.py`) and the Streamlit
demo (`src/deploy/app.py`).

The encoder/architecture is inferred per the order:
1. **Hydra training dump** at `outputs/<run_name>/.hydra/config.yaml`, when
   present — gives the authoritative `ModelConfig` (incl. `use_ctc`,
   `rnn_hidden_dim`, layer counts). This is how CRNN+CTC runs are detected,
   since they live under `resnet-<timestamp>/` directories.
2. Falling back to parsing the leading token of the checkpoint dir name
   (`vit-...` / `resnet-...`) plus a hardcoded default `ModelConfig`.

Both paths can be overridden with the `encoder=` / `model_config=` kwargs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file

from src.data import build_default_vocabs, get_encoder_spec
from src.data.vocabulary import Vocabulary
from src.model import ModelConfig, OMRModel
from src.model.config import LossWeights, MaskNullInLoss
from src.model.encoders import build_encoder
from src.postproc import (
    JianpuRenderConfig,
    TokenTuple,
    ids_to_jianpu,
    ids_to_tuples,
)

_STREAMS = ("type", "pitch", "rhythm", "attribute")

ImageInput = Image.Image | np.ndarray | str | Path

# Architecture of the released AR checkpoints (configs/model/{vit,resnet}.yaml).
# Used only when no Hydra training dump is found alongside the checkpoint.
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


def _find_hydra_config(ckpt_dir: Path) -> Path | None:
    """Locate the Hydra dump that recorded this checkpoint's training config.

    The training runner writes to `outputs/<run_name>/.hydra/config.yaml` where
    `run_name = "<encoder>-<timestamp>"` (see configs/train.yaml). The
    checkpoint lives at `checkpoints/<run_name>/step-N-best/` (or
    `checkpoints/<run_name>/`), so walk up to find the run-name component and
    then check the sibling `outputs/<run_name>/.hydra/config.yaml`.
    """
    parts: list[Path] = [ckpt_dir, *ckpt_dir.parents]
    repo_roots: set[Path] = set()
    for part in parts:
        # The repo root is two levels above `checkpoints/<run>`; walk up to find
        # any ancestor that has an `outputs/` sibling.
        for ancestor in [part, *part.parents]:
            if (ancestor / "outputs").is_dir():
                repo_roots.add(ancestor)
                break
    run_name_candidates: list[str] = []
    for part in parts:
        name = part.name
        if name and name != ".." and not _STEP_BEST_RE.fullmatch(name):
            run_name_candidates.append(name)
    for root in repo_roots:
        for run_name in run_name_candidates:
            cfg = root / "outputs" / run_name / ".hydra" / "config.yaml"
            if cfg.is_file():
                return cfg
    return None


def _model_config_from_hydra(hydra_cfg: dict[str, Any]) -> tuple[ModelConfig, str]:
    """Build a `ModelConfig` and resolved encoder name from a Hydra dump."""
    m = hydra_cfg.get("model", {}) or {}
    encoder_name = m.get("encoder_name", "vit")
    cfg = ModelConfig(
        d_model=m.get("d_model", 384),
        decoder_layers=m.get("decoder_layers", 4),
        decoder_heads=m.get("decoder_heads", 6),
        decoder_ffn_dim=m.get("decoder_ffn_dim", 1536),
        dropout=m.get("dropout", 0.1),
        max_decoder_positions=m.get("max_decoder_positions", 512),
        scale_embedding=m.get("scale_embedding", True),
        eos_weight=m.get("eos_weight", 1.0),
        loss_weights=LossWeights(**(m.get("loss_weights") or {})),
        mask_null_in_loss=MaskNullInLoss(**(m.get("mask_null_in_loss") or {})),
        use_ctc=m.get("use_ctc", False),
        rnn_hidden_dim=m.get("rnn_hidden_dim", 256),
        rnn_bidirectional=m.get("rnn_bidirectional", True),
    )
    return cfg, encoder_name


_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")


def _has_weights(d: Path) -> bool:
    return any((d / name).exists() for name in _WEIGHT_FILENAMES)


def _load_weights(weights_dir: Path) -> dict:
    """Load whichever weight format the checkpoint uses.

    accelerator.save_state writes `model.safetensors` by default and
    `pytorch_model.bin` when `safe_serialization=False`. CRNN+CTC checkpoints
    must use the latter (PyTorch LSTM's `_flat_weights` aliasing breaks safetensors).
    """
    sf = weights_dir / "model.safetensors"
    if sf.exists():
        return load_file(str(sf))
    bin_ = weights_dir / "pytorch_model.bin"
    if bin_.exists():
        return torch.load(str(bin_), map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"no model.safetensors or pytorch_model.bin in {weights_dir}"
    )


def _resolve_checkpoint(ckpt_dir: Path) -> Path:
    """Return the directory holding `model.safetensors` or `pytorch_model.bin`.

    Accepts either a leaf checkpoint dir or a run dir containing many
    `step-N-best/` subdirs; in the latter case picks the highest step (the
    best val-SER snapshot saved last).
    """
    if _has_weights(ckpt_dir):
        return ckpt_dir
    candidates = [
        (int(m.group(1)), p)
        for p in ckpt_dir.iterdir()
        if p.is_dir() and (m := _STEP_BEST_RE.fullmatch(p.name))
    ]
    if not candidates:
        raise FileNotFoundError(
            "no model.safetensors / pytorch_model.bin and no step-N-best/ "
            f"subdir under {ckpt_dir}"
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

        # 1) Prefer the Hydra training dump if present (authoritative for
        #    architecture, especially CRNN+CTC runs which live under
        #    `resnet-*` dirs but use `MultiHeadCTCDecoder`).
        hydra_path = _find_hydra_config(ckpt_dir)
        hydra_cfg: dict[str, Any] | None = None
        if hydra_path is not None:
            try:
                hydra_cfg = yaml.safe_load(hydra_path.read_text()) or {}
            except yaml.YAMLError:
                hydra_cfg = None

        if model_config is not None:
            resolved_cfg = model_config
            resolved_encoder = encoder or _infer_encoder_name(ckpt_dir)
        elif hydra_cfg is not None:
            resolved_cfg, hydra_encoder = _model_config_from_hydra(hydra_cfg)
            resolved_encoder = encoder or hydra_encoder
        else:
            resolved_cfg = DEFAULT_MODEL_CONFIG
            resolved_encoder = encoder or _infer_encoder_name(ckpt_dir)

        self.encoder_name = resolved_encoder
        self.model_config = resolved_cfg
        self.use_ctc = resolved_cfg.use_ctc
        self.hydra_config_path = hydra_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.vocabs = build_default_vocabs()
        self.spec = get_encoder_spec(self.encoder_name)
        self.transform = self.spec.build_eval_transform()

        # Only ViT downloads pretrained weights; build the bare arch and let the
        # checkpoint supply the trained weights. ResNet has no such kwarg.
        build_kwargs = {"pretrained": False} if self.encoder_name == "vit" else {}
        encoder_mod = build_encoder(self.spec, resolved_cfg.d_model, **build_kwargs)
        self.model = OMRModel(encoder=encoder_mod, vocabs=self.vocabs, cfg=resolved_cfg)
        weights_dir = _resolve_checkpoint(ckpt_dir)
        state = _load_weights(weights_dir)
        missing, _ = self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        self.checkpoint_path = weights_dir

        # safetensors + nn.LSTM truncation: PyTorch LSTM's `_flat_weights` are
        # views into a single contiguous buffer, so `state_dict()` produces 16+
        # tensors that all share storage. safetensors refuses to store aliasing
        # tensors and silently keeps only one (`weight_ih_l0`), losing the rest.
        # In-memory training stays correct (hence "val SER 0"), but on disk the
        # LSTM is effectively destroyed — at load time we get a randomly-init'd
        # LSTM and garbage predictions. Detect this clearly so it doesn't show
        # up as a silent accuracy collapse.
        if self.use_ctc:
            lstm_missing = [k for k in missing if "rnn" in k]
            if lstm_missing:
                raise RuntimeError(
                    f"CRNN+CTC checkpoint at {weights_dir} is missing "
                    f"{len(lstm_missing)} LSTM parameters (e.g. {lstm_missing[0]!r}). "
                    "This is the known safetensors+nn.LSTM aliasing bug: the "
                    "LSTM's flat-weights share storage and safetensors stored "
                    "only one of them. The checkpoint cannot be deployed as-is. "
                    "Fix in training: pass safe_serialization=False to "
                    "accelerator.save_state(), or clone the state_dict before "
                    "save (`{k: v.clone() for k,v in model.state_dict().items()}`)."
                )

    @property
    def architecture(self) -> str:
        """Human-readable arch flavour for UI/CLI ('ResNet+CRNN+CTC' etc.)."""
        head = "CRNN+CTC" if self.use_ctc else "AR multi-head"
        encoder = "ViT" if self.encoder_name == "vit" else "ResNet"
        return f"{encoder} + {head}"

    def _ctc_decode_aligned(
        self, logits: dict[str, torch.Tensor]
    ) -> list[dict[str, list[int]]]:
        """Type-anchored CTC greedy collapse → row-aligned id streams per sample.

        The model's own `generate()` collapses each head independently, which
        destroys cross-head alignment (postproc expects `(type[i], pitch[i],
        rhythm[i], attr[i])` to describe one symbol). Here we use the type
        stream as the structural authority: walk the encoder time axis, emit
        a row only when type's argmax is non-blank and not a duplicate of the
        previous emitted type; at each emission, pull pitch/rhythm/attribute
        from the **same** time step. Blank in a non-type head maps to NULL_ID
        (matching the training-time NULL semantics).
        """
        blanks = self.model.decoder.blank_ids  # name -> blank id (= len(vocab))
        B = logits["type"].shape[0]
        out: list[dict[str, list[int]]] = []
        for b in range(B):
            argmax = {n: logits[n][b].argmax(dim=-1).tolist() for n in _STREAMS}
            streams: dict[str, list[int]] = {n: [] for n in _STREAMS}
            prev_type: int | None = None
            for t in range(len(argmax["type"])):
                tt = argmax["type"][t]
                if tt == blanks["type"]:
                    prev_type = None
                    continue
                if tt == prev_type:
                    continue
                # Truncate at EOS in the type stream (matches AR contract).
                if tt == Vocabulary.EOS_ID:
                    break
                # PAD/BOS at a non-blank type position is a model hallucination;
                # skip the row, don't propagate to postproc.
                if tt in (Vocabulary.PAD_ID, Vocabulary.BOS_ID):
                    prev_type = tt
                    continue
                streams["type"].append(tt)
                for n in ("pitch", "rhythm", "attribute"):
                    vid = argmax[n][t]
                    if vid == blanks[n]:
                        vid = Vocabulary.NULL_ID
                    streams[n].append(vid)
                prev_type = tt
            out.append(streams)
        return out

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

        # CTC mode: run forward and do type-anchored CTC collapse so the four
        # heads stay row-aligned (the model's own `generate()` collapses each
        # head independently). AR mode: the per-row alignment is intrinsic, so
        # use the model's KV-cached greedy generate.
        if self.use_ctc:
            out = self.model.forward(pixel_values)
            streams = self._ctc_decode_aligned(out["logits"])
            per_sample: list[tuple[list[int], list[int], list[int], list[int]]] = [
                (s["type"], s["pitch"], s["rhythm"], s["attribute"]) for s in streams
            ]
        else:
            gen = self.model.generate(pixel_values, max_length=self.max_length)
            per_sample = []
            for i in range(len(images)):
                length = int(gen.lengths[i].item())
                per_sample.append(
                    (
                        gen.type_ids[i][:length].tolist(),
                        gen.pitch_ids[i][:length].tolist(),
                        gen.rhythm_ids[i][:length].tolist(),
                        gen.attribute_ids[i][:length].tolist(),
                    )
                )

        results: list[JianpuPrediction] = []
        for t, p, r, a in per_sample:
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
                    length=len(t),
                )
            )
        return results

    def predict(self, image: ImageInput) -> JianpuPrediction:
        return self.predict_batch([image])[0]
