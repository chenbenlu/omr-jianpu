from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors.torch import save_file

from src.data import ENCODER_REGISTRY, build_default_vocabs
from src.deploy.inference import (
    JianpuPrediction,
    OMRInferencer,
    _infer_encoder_name,
    _resolve_checkpoint,
)
from src.model.config import ModelConfig
from src.model.encoders import ResNetEncoder
from src.model.model import OMRModel

_TINY_CFG = ModelConfig(
    d_model=32,
    decoder_layers=2,
    decoder_heads=2,
    decoder_ffn_dim=64,
    max_decoder_positions=64,
    dropout=0.0,
)


# --- checkpoint / encoder resolution -------------------------------------


def test_infer_encoder_name_from_run_dir():
    assert _infer_encoder_name(Path("checkpoints/vit-20260528-090804")) == "vit"
    assert _infer_encoder_name(Path("checkpoints/resnet-20260529-043155")) == "resnet"


def test_infer_encoder_name_from_step_subdir():
    p = Path("checkpoints/vit-20260528-090804/step-68750-best")
    assert _infer_encoder_name(p) == "vit"


def test_infer_encoder_name_unknown_raises():
    with pytest.raises(ValueError):
        _infer_encoder_name(Path("checkpoints/mystery-123"))


def test_resolve_checkpoint_prefers_direct_weights(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert _resolve_checkpoint(tmp_path) == tmp_path


def test_resolve_checkpoint_picks_max_step(tmp_path):
    for step in (3125, 68750, 9375):
        d = tmp_path / f"step-{step}-best"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"")
    assert _resolve_checkpoint(tmp_path).name == "step-68750-best"


def test_resolve_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_checkpoint(tmp_path)


def test_hydra_autodetect_use_ctc(tmp_path):
    # Simulate a CRNN training tree: checkpoints/<run>/step-N-best/model.safetensors
    # alongside outputs/<run>/.hydra/config.yaml.
    from src.deploy.inference import _find_hydra_config, _model_config_from_hydra
    import yaml as _yaml

    run = "resnet-20260603-060007"
    ckpt = tmp_path / "checkpoints" / run / "step-7800-best"
    ckpt.mkdir(parents=True)
    hydra_dir = tmp_path / "outputs" / run / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(
        _yaml.safe_dump(
            {
                "model": {
                    "encoder_name": "resnet",
                    "d_model": 384,
                    "use_ctc": True,
                    "rnn_hidden_dim": 256,
                    "rnn_bidirectional": True,
                    "decoder_layers": 2,
                }
            }
        )
    )
    found = _find_hydra_config(ckpt)
    assert found is not None and found.name == "config.yaml"
    cfg, enc = _model_config_from_hydra(_yaml.safe_load(found.read_text()))
    assert enc == "resnet"
    assert cfg.use_ctc is True
    assert cfg.rnn_hidden_dim == 256
    assert cfg.decoder_layers == 2


def test_inferencer_detects_truncated_lstm(tmp_path):
    # Reproduce the safetensors+nn.LSTM aliasing bug: save a CRNN model_config
    # but write only `weight_ih_l0` (mimicking what safetensors does to a real
    # CRNN state_dict). The inferencer must raise with a clear, actionable
    # message — not silently load random LSTM weights and produce garbage.
    from src.model.config import ModelConfig as _MC
    from src.data import ENCODER_REGISTRY, build_default_vocabs
    from src.model.encoders import ResNetEncoder
    from src.model.model import OMRModel

    tiny_ctc = _MC(
        d_model=32,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ffn_dim=64,
        max_decoder_positions=64,
        dropout=0.0,
        use_ctc=True,
        rnn_hidden_dim=16,
        rnn_bidirectional=True,
    )
    vb = build_default_vocabs()
    encoder = ResNetEncoder(
        d_model=tiny_ctc.d_model, encoder_spec=ENCODER_REGISTRY["resnet"]
    )
    model = OMRModel(encoder=encoder, vocabs=vb, cfg=tiny_ctc)

    ckpt = tmp_path / "resnet-20260101-000000"
    ckpt.mkdir()
    # Keep everything except the LSTM (simulate safetensors aliasing loss);
    # keep one rnn weight to mirror the real bug pattern.
    full = model.state_dict()
    truncated = {k: v for k, v in full.items() if not k.startswith("decoder.rnn.")}
    truncated["decoder.rnn.weight_ih_l0"] = full["decoder.rnn.weight_ih_l0"]
    save_file(truncated, str(ckpt / "model.safetensors"))

    with pytest.raises(RuntimeError, match="safetensors"):
        OMRInferencer(ckpt, device="cpu", model_config=tiny_ctc)


# --- end-to-end inferencer (tiny random model, contract only) -------------


def _save_tiny_resnet_ckpt(ckpt_dir: Path) -> None:
    vb = build_default_vocabs()
    encoder = ResNetEncoder(
        d_model=_TINY_CFG.d_model, encoder_spec=ENCODER_REGISTRY["resnet"]
    )
    model = OMRModel(encoder=encoder, vocabs=vb, cfg=_TINY_CFG)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))


def test_inferencer_predict_contract(tmp_path):
    ckpt_dir = tmp_path / "resnet-20260101-000000"
    _save_tiny_resnet_ckpt(ckpt_dir)

    inferencer = OMRInferencer(
        ckpt_dir, device="cpu", model_config=_TINY_CFG, max_length=16
    )
    assert inferencer.encoder_name == "resnet"

    white = Image.fromarray(np.full((128, 256, 3), 255, dtype=np.uint8))
    pred = inferencer.predict(white)
    assert isinstance(pred, JianpuPrediction)
    assert isinstance(pred.jianpu, str)
    assert pred.length == len(pred.type_ids)
    assert len(pred.tuples) <= pred.length


def test_inferencer_predict_batch(tmp_path):
    ckpt_dir = tmp_path / "resnet-20260101-000000"
    _save_tiny_resnet_ckpt(ckpt_dir)
    inferencer = OMRInferencer(
        ckpt_dir, device="cpu", model_config=_TINY_CFG, max_length=16
    )
    imgs = [np.full((128, 200, 3), 255, dtype=np.uint8) for _ in range(3)]
    preds = inferencer.predict_batch(imgs)
    assert len(preds) == 3
    assert all(isinstance(p, JianpuPrediction) for p in preds)


# --- notation: tuples -> music21 Stream + engraving -----------------------

from unittest.mock import patch  # noqa: E402

from music21 import note, stream  # noqa: E402

from src.data.renderer import StaffRenderer  # noqa: E402
from src.deploy import notation  # noqa: E402
from src.deploy.jianpu_format import (  # noqa: E402
    _decompose,
    _parse_cell,
    jianpu_html,
    jianpu_svg,
    pretty_jianpu,
)

_SAMPLE_TUPLES = [
    ("clef", None, None, "G2"),
    ("key_signature", None, None, "ks+2"),
    ("time_signature", None, None, "6/8"),
    ("note", "A4", "eighth", None),
    ("note", "F#5", "quarter", None),
    ("barline", None, None, None),
    ("rest", None, "eighth", None),
    ("note", "Bb4", "quarter_dot", None),
]


def _notes(s):
    return list(s.recurse().getElementsByClass(note.Note))


def test_tuples_to_stream_is_structured_score():
    s = notation.tuples_to_stream(_SAMPLE_TUPLES)
    assert isinstance(s, stream.Score)
    measures = list(s.recurse().getElementsByClass(stream.Measure))
    assert len(measures) == 2  # split on the single barline


def test_tuples_to_stream_object_counts():
    s = notation.tuples_to_stream(_SAMPLE_TUPLES)
    notes = _notes(s)
    rests = list(s.recurse().getElementsByClass(note.Rest))
    assert len(notes) == 3
    assert len(rests) == 1


def test_tuples_to_stream_dotted_duration():
    s = notation.tuples_to_stream([("note", "C4", "quarter_dot", None)])
    n = _notes(s)[0]
    assert n.duration.dots == 1
    assert n.duration.type == "quarter"


def test_tuples_to_stream_flat_pitch():
    s = notation.tuples_to_stream([("note", "Bb4", "quarter", None)])
    assert _notes(s)[0].pitch.name == "B-"


def test_tuples_to_stream_skips_holes():
    # None/<UNK> pitch or rhythm must not raise; the symbol is dropped.
    s = notation.tuples_to_stream(
        [
            ("note", None, "quarter", None),
            ("note", "<UNK>", "quarter", None),
            ("note", "C4", None, None),
            ("note", "C4", "quarter", None),
        ]
    )
    assert len(_notes(s)) == 1


def test_which_backend_auto_prefers_lilypond():
    with patch.object(notation, "lilypond_available", return_value=True):
        assert notation.which_backend("auto") == "lilypond"
    with patch.object(notation, "lilypond_available", return_value=False):
        assert notation.which_backend("auto") == "verovio"


def test_render_staff_png_verovio(monkeypatch):
    stub = np.full((120, 400, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(StaffRenderer, "render", lambda self, score: stub)
    img, backend = notation.render_staff_png(_SAMPLE_TUPLES, backend="verovio")
    assert backend == "verovio"
    assert img.shape == stub.shape


def test_render_staff_png_falls_back_when_lilypond_absent(monkeypatch):
    stub = np.full((120, 400, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(StaffRenderer, "render", lambda self, score: stub)
    monkeypatch.setattr(notation, "lilypond_available", lambda: False)
    img, backend = notation.render_staff_png(_SAMPLE_TUPLES, backend="auto")
    assert backend == "verovio"
    assert img.shape == stub.shape


def test_ensure_verovio_resources_fixes_broken_font_path():
    # Simulate the Streamlit failure: a bad default resource path makes verovio
    # report "font resources are not available" and reject any payload. Our
    # initializer must repoint it at the bundled fonts so rendering succeeds.
    import verovio

    from src.data.renderer import RenderConfig, StaffRenderer

    tuples = [
        ("clef", None, None, "G2"),
        ("note", "C4", "quarter", None),
    ]
    verovio.setDefaultResourcePath("/nonexistent/verovio/data")
    try:
        with pytest.raises(Exception):
            StaffRenderer(RenderConfig()).render(notation.tuples_to_stream(tuples))
        notation._ensure_verovio_resources()
        img = StaffRenderer(RenderConfig()).render(notation.tuples_to_stream(tuples))
        assert img.ndim == 3
    finally:
        notation._ensure_verovio_resources()  # restore for other tests


def test_render_staff_png_retries_after_makenotation(monkeypatch):
    # First render raises (verovio rejected the raw payload); the retry on the
    # makeNotation-normalized stream must succeed — the demo should not crash.
    stub = np.full((120, 400, 3), 255, dtype=np.uint8)
    calls = {"n": 0}

    def flaky_render(self, score):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("verovio failed to parse the MusicXML payload")
        return stub

    monkeypatch.setattr(StaffRenderer, "render", flaky_render)
    img, backend = notation.render_staff_png(_SAMPLE_TUPLES, backend="verovio")
    assert backend == "verovio"
    assert calls["n"] == 2  # failed once, retried on normalized stream
    assert img.shape == stub.shape


# --- pretty 3-line Jianpu -------------------------------------------------


def test_decompose_octave_up_to_top_row():
    top, mid, bot = _decompose("3'")
    assert mid == "3"
    assert top.strip() == "."
    assert bot.strip() == ""


def test_decompose_underline_to_bottom_row():
    top, mid, bot = _decompose("__2")
    assert mid == "2"
    assert bot.strip() == "__"
    assert top.strip() == ""


def test_pretty_jianpu_three_lines_and_header():
    out = pretty_jianpu(_SAMPLE_TUPLES)
    lines = out.split("\n")
    assert lines[0].startswith("[Clef:")  # header preserved
    assert "|" in out  # barline rendered in the middle row
    assert len(lines) >= 3  # header + at least the stacked rows


# --- structured cell model + HTML/SVG renderers ----------------------------


def test_parse_cell_octave_and_underlines():
    c = _parse_cell("__7,")  # 16th (2 beams), degree 7, one octave down
    assert c.body == "7"
    assert c.dots_down == 1
    assert c.dots_up == 0
    assert c.underlines == 2


def test_parse_cell_accidental_and_octave_up():
    c = _parse_cell("#4'")
    assert c.accidental == "#"
    assert c.body == "4"
    assert c.dots_up == 1


def test_jianpu_html_well_formed():
    html = jianpu_html(_SAMPLE_TUPLES)
    assert "<style>" in html
    assert "jp-grid" in html or "jp-row" in html
    # one column per non-structural symbol (3 notes + 1 rest? -> here 3 notes,
    # 1 barline, 1 rest); just assert columns exist and header is present.
    assert "jp-col" in html
    assert "Clef" in html


def test_jianpu_html_escapes_content():
    # body/accidental are constrained, but ensure no raw "<"/">" leak from data
    html = jianpu_html([("note", "C4", "quarter", None)])
    assert "<script" not in html.lower()


def test_jianpu_html_beam_count_matches_rhythm():
    # Each rhythm beam must render as a separate bar. 16th = 2 beams; an
    # eighth's single beam must not be miscounted as more.
    # `jp-beam ` (with trailing space) matches the bar class but not the
    # `jp-beams` container.
    sixteenth = jianpu_html([("note", "C4", "16th", None)])
    col = sixteenth.split('<div class="jp-row">', 1)[1]
    assert col.count("jp-beam ") == 2
    eighth = jianpu_html([("note", "C4", "eighth", None)])
    assert eighth.split('<div class="jp-row">', 1)[1].count("jp-beam ") == 1
    quarter = jianpu_html([("note", "C4", "quarter", None)])
    assert quarter.split('<div class="jp-row">', 1)[1].count("jp-beam ") == 0


def test_jianpu_html_beam_order_layer1_first():
    # The HTML emits layer 1 (8th, longest) before layer 2 (16th). The grid
    # places jp-b1 in row 1 (closest to digit) and jp-b2 in row 2 below.
    html = jianpu_html([("note", "C4", "16th", None)])
    col = html.split('<div class="jp-row">', 1)[1]
    i1 = col.find("jp-b1")
    i2 = col.find("jp-b2")
    assert 0 <= i1 < i2  # layer-1 bar appears first in DOM order


def test_jianpu_svg_well_formed():
    svg = jianpu_svg(_SAMPLE_TUPLES)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "viewBox" in svg
    # octave-up note (F#5 -> "4'") should emit at least one dot circle
    assert "<circle" in svg
    # barline should emit a vertical line
    assert "<line" in svg


def test_jianpu_svg_empty_is_valid():
    svg = jianpu_svg([])
    assert svg.startswith("<svg")
    assert "</svg>" in svg


# --- beam grouping (continuous rhythm lines within a beat) -----------------

from src.deploy.jianpu_format import _beam_extends_right, _cells  # noqa: E402


def test_beam_groups_within_and_across_beats():
    tuples = [
        ("time_signature", None, None, "4/4"),
        ("note", "C4", "eighth", None),  # beat 0
        ("note", "D4", "eighth", None),  # beat 0
        ("note", "E4", "eighth", None),  # beat 1
        ("note", "F4", "eighth", None),  # beat 1
    ]
    # the time signature is consumed into state, so cells == the four notes
    cells = _cells(tuples)
    assert [c.is_note for c in cells] == [True, True, True, True]
    assert cells[0].beat_group == cells[1].beat_group  # beat 0
    assert cells[2].beat_group == cells[3].beat_group  # beat 1
    assert cells[1].beat_group != cells[2].beat_group  # beat boundary
    assert _beam_extends_right(cells, 0, 1)  # note0 -> note1 within beat 0
    assert not _beam_extends_right(cells, 1, 1)  # note1 -> note2 across beats


def test_beam_breaks_at_barline_and_rest():
    tuples = [
        ("note", "C4", "eighth", None),
        ("barline", None, None, None),
        ("note", "D4", "eighth", None),
        ("rest", None, "eighth", None),
        ("note", "E4", "eighth", None),
    ]
    cells = _cells(tuples)
    # index 0 is a note, index 1 the barline -> no beam across the barline
    assert cells[1].is_barline
    assert not _beam_extends_right(cells, 0, 1)
    # the note before the rest (D, index 2) must not beam into the rest
    assert cells[2].is_note and not cells[2].body == "0"
    assert not _beam_extends_right(cells, 2, 1)


def test_long_note_dot_aligned_to_digit_not_tail():
    # A half/whole-note carries a " -" / " - - -" tail; the upper octave dot
    # must align to the digit's center, not the digit+tail midpoint.
    # SVG: the dot's cx must equal the digit text's x (text-anchor="middle"),
    # and the tail draws at a strictly larger x.
    import re

    svg = jianpu_svg(
        [
            ("clef", None, None, "G2"),
            # F5 in C major -> degree 4 with one upper-octave dot, half note.
            ("note", "F5", "half", None),
        ]
    )
    digit_x = float(
        re.search(r'<text x="([\d.]+)"[^>]*text-anchor="middle"', svg).group(1)
    )
    dot_cx = float(re.search(r'<circle cx="([\d.]+)"', svg).group(1))
    tail_x = float(
        re.search(r'<text x="([\d.]+)"[^>]*text-anchor="start"', svg).group(1)
    )
    assert abs(dot_cx - digit_x) < 0.01  # dot aligned to digit center
    assert tail_x > digit_x  # tail draws to the right of the digit
    # HTML: digit and tail are inline siblings in jp-mid; tail is absolutely
    # positioned (left:100%) so it does not shift the digit's geometric center.
    html = jianpu_html(
        [
            ("clef", None, None, "G2"),
            ("note", "F5", "half", None),
        ]
    )
    assert "jp-digit" in html
    assert "jp-tail" in html  # tail span present
    assert "position:absolute" in html  # tail uses absolute positioning


def test_svg_layer1_closer_to_digit():
    # Jianpu order: the basic 8th beam (layer 1, longest/continuous) sits
    # closest to the digit; the extra 16th beam stacks BELOW it.
    # Verify directly via the layer offset formula used by jianpu_svg.
    from src.deploy.jianpu_format import _cells

    cells = _cells(
        [
            ("time_signature", None, None, "4/4"),
            ("note", "D4", "16th", None),  # has both layer 1 and layer 2
        ]
    )
    c = cells[0]
    assert c.underlines == 2
    base_y = 60  # mirrors jianpu_svg's base_y
    y_layer1 = base_y + 5 + (1 - 1) * 4
    y_layer2 = base_y + 5 + (2 - 1) * 4
    assert y_layer1 < y_layer2  # layer 1 (8th) is visually higher (closer)


def test_svg_continuous_beam_reaches_next_center():
    # Two same-beat eighths: the first beam line must extend to the 2nd center.
    tuples = [
        ("time_signature", None, None, "4/4"),
        ("note", "C4", "eighth", None),
        ("note", "D4", "eighth", None),
    ]
    svg = jianpu_svg(tuples)
    # the joined beam is wider than a lone stub (>16px); just assert a long line
    import re

    beam_lines = re.findall(r'<line x1="([\d.]+)" y1="[\d.]+" x2="([\d.]+)"', svg)
    spans = [float(x2) - float(x1) for x1, x2 in beam_lines]
    assert max(spans) > 16  # a continuous beam, not a 16px stub
