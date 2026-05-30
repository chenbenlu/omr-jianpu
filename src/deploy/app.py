"""Streamlit demo: OMR staff image -> Jianpu.

Run:
    streamlit run src/deploy/app.py

Wraps `src.deploy.OMRInferencer` (no inference logic is duplicated here). Lets
you upload a staff PNG or pick a pre-rendered val sample; for val samples the
ground-truth Jianpu is shown beside the prediction for a live correctness check.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.deploy import (
    OMRInferencer,
    jianpu_html,
    lilypond_available,
    render_staff_png,
)
from src.postproc import JianpuRenderConfig, ids_to_jianpu

# Spaces sets SPACE_ID; treat that as the single signal that we're running
# on Hugging Face Spaces (pull checkpoint from the Hub, hide dev-only sidebar
# controls, read samples from the bundled `samples/` dir instead of the full
# pre-rendered val split).
ON_SPACES = bool(os.environ.get("SPACE_ID"))

CKPT_ROOT = Path("checkpoints")
# On Spaces the bundled samples live next to the Space repo root, two levels
# up from this file (src/deploy/app.py -> repo/).
SPACES_SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
LOCAL_VAL_DIR = Path("data/synthetic/val")
VAL_DIR = SPACES_SAMPLES_DIR if ON_SPACES else LOCAL_VAL_DIR
VAL_MANIFEST_NAME = "sample_manifest.jsonl" if ON_SPACES else "manifest.jsonl"

# Default model repo used on Spaces; overridable via env so a fork can point
# at its own weights without editing source.
DEFAULT_MODEL_REPO = "chenbenlu/omr-to-jianpu-vit"


@st.cache_resource(show_spinner="Downloading model from Hugging Face Hub…")
def resolve_spaces_checkpoint() -> str:
    """Pull the ViT checkpoint to the Spaces persistent cache once per cold start.

    `OMR_LOCAL_CKPT` short-circuits to a local path — used by the pre-push
    smoke test so we exercise the Spaces code path without round-tripping
    through the Hub.
    """
    local_override = os.environ.get("OMR_LOCAL_CKPT")
    if local_override:
        return local_override
    from huggingface_hub import snapshot_download

    repo_id = os.environ.get("OMR_MODEL_REPO", DEFAULT_MODEL_REPO)
    return snapshot_download(repo_id=repo_id)


@st.cache_resource(show_spinner="Loading model…")
def load_inferencer(ckpt_dir: str, encoder: str, max_length: int) -> OMRInferencer:
    return OMRInferencer(
        ckpt_dir,
        encoder=None if encoder == "auto" else encoder,
        max_length=max_length,
    )


@st.cache_data
def load_val_manifest() -> list[dict]:
    path = VAL_DIR / VAL_MANIFEST_NAME
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


def ground_truth_jianpu(row: dict, inferencer: OMRInferencer) -> str:
    vb = inferencer.vocabs
    return ids_to_jianpu(
        vb.type.encode(row["type"]),
        vb.pitch.encode(row["pitch"]),
        vb.rhythm.encode(row["rhythm"]),
        vb.attribute.encode(row["attribute"]),
        vb,
        JianpuRenderConfig(emit_header=True),
    )


def render_prediction(
    image: Image.Image, inferencer: OMRInferencer, gt: str | None, view: str
) -> None:
    start = time.perf_counter()
    with st.spinner("Predicting… (CPU decode ~10–30 s on Spaces)"):
        pred = inferencer.predict(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    left, right = st.columns(2)
    with left:
        st.subheader("Input")
        st.image(image, width="stretch")
    with right:
        if view == "Engraved":
            st.subheader("Engraved (predicted)")
            try:
                staff, backend = render_staff_png(pred.tuples)
                st.image(staff, width="stretch")
                st.caption(
                    f"engraver: {backend} · {pred.length} symbols · {elapsed_ms:.0f} ms"
                )
            except Exception as exc:
                st.error(f"Engraving failed: {exc}")
        else:
            st.subheader("Predicted Jianpu")
            if view == "Jianpu (pretty)":
                st.markdown(jianpu_html(pred.tuples), unsafe_allow_html=True)
            else:
                st.code(pred.jianpu, language=None)
            st.caption(f"{pred.length} symbols · {elapsed_ms:.0f} ms")
        if gt is not None:
            st.subheader("Ground truth (Jianpu)")
            st.code(gt, language=None)
            st.success("Match") if gt == pred.jianpu else st.warning("Differs from GT")

    with st.expander("Per-symbol decode (type, pitch, rhythm, attribute)"):
        st.dataframe(
            pd.DataFrame(pred.tuples, columns=["type", "pitch", "rhythm", "attribute"]),
            width="stretch",
        )


def main() -> None:
    st.set_page_config(page_title="OMR → Jianpu", layout="wide")
    st.title("OMR → Jianpu Demo")
    st.caption("Printed monophonic staff notation → structured Jianpu text.")

    with st.sidebar:
        st.header("Model")
        if ON_SPACES:
            ckpt_dir = resolve_spaces_checkpoint()
            encoder = "vit"
            max_length = 64
            st.caption("ViT encoder · checkpoint pulled from the HF Hub")
        else:
            ckpt_dirs = sorted(p.name for p in CKPT_ROOT.glob("*") if p.is_dir())
            if ckpt_dirs:
                ckpt_name = st.selectbox(
                    "Checkpoint", ckpt_dirs, index=len(ckpt_dirs) - 1
                )
                ckpt_dir = str(CKPT_ROOT / ckpt_name)
            else:
                ckpt_dir = st.text_input("Checkpoint dir", value=str(CKPT_ROOT))
            encoder = st.selectbox("Encoder", ["auto", "vit", "resnet"])
            max_length = st.slider("Max decode length", 16, 256, 64, step=16)
        st.header("Output")
        view = st.radio(
            "Notation view",
            ["Engraved", "Jianpu (compact ASCII)", "Jianpu (pretty)"],
        )
        if view == "Engraved":
            engraver = "lilypond" if lilypond_available() else "verovio"
            st.caption(f"engraver: {engraver}")
            if engraver == "verovio":
                st.info(
                    "Engraving uses verovio. Install `lilypond` on PATH to "
                    "switch to LilyPond engraving automatically."
                )

    try:
        inferencer = load_inferencer(ckpt_dir, encoder, max_length)
    except Exception as exc:  # surface load errors in-page instead of a stack trace
        st.error(f"Failed to load checkpoint: {exc}")
        return
    st.sidebar.success(f"Loaded `{inferencer.encoder_name}` on `{inferencer.device}`")
    if inferencer.encoder_name == "resnet":
        st.sidebar.info(
            "ResNet encoder cannot learn pitch (documented structural "
            "limitation); expect correct rhythm but wrong pitch numbers."
        )

    upload_tab, sample_tab = st.tabs(["Upload image", "Val sample"])

    with upload_tab:
        file = st.file_uploader("Staff image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if file is not None:
            render_prediction(Image.open(file), inferencer, gt=None, view=view)

    with sample_tab:
        manifest = load_val_manifest()
        if not manifest:
            st.info(f"No manifest at {VAL_DIR / VAL_MANIFEST_NAME}.")
        else:
            idx = st.number_input("Sample index", 0, len(manifest) - 1, 0, step=1)
            row = manifest[int(idx)]
            image = Image.open(VAL_DIR / row["image"])
            render_prediction(
                image, inferencer, gt=ground_truth_jianpu(row, inferencer), view=view
            )


if __name__ == "__main__":
    main()
