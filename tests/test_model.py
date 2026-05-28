from __future__ import annotations

import torch

from src.data import ENCODER_REGISTRY, Vocabulary, build_default_vocabs
from src.model.config import MaskNullInLoss, ModelConfig
from src.model.decoder import MultiHeadDecoder
from src.model.encoders import ResNetEncoder, ViTEncoder
from src.model.losses import compute_loss
from src.model.model import OMRModel

_TINY_DECODER_CFG = ModelConfig(
    d_model=32,
    decoder_layers=2,
    decoder_heads=2,
    decoder_ffn_dim=64,
    max_decoder_positions=64,
    dropout=0.0,
)


def _make_ids(vb, B: int, L: int) -> dict[str, torch.Tensor]:
    return {
        "type": torch.randint(0, len(vb.type), (B, L)),
        "pitch": torch.randint(0, len(vb.pitch), (B, L)),
        "rhythm": torch.randint(0, len(vb.rhythm), (B, L)),
        "attribute": torch.randint(0, len(vb.attribute), (B, L)),
    }


_TINY_VIT_CFG = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
}


def test_vit_encoder_drops_cls() -> None:
    encoder = ViTEncoder(d_model=32, pretrained=False, vit_config_kwargs=_TINY_VIT_CFG)
    encoder.eval()
    out = encoder(torch.randn(2, 3, 224, 224))
    # 14*14 patches = 196; [CLS] gives raw 197, we drop it → 196.
    assert out.hidden_states.shape == (2, 196, 32)
    assert out.attention_mask is None


def test_resnet_encoder_shape() -> None:
    encoder = ResNetEncoder(d_model=32, encoder_spec=ENCODER_REGISTRY["resnet"])
    encoder.eval()
    out1 = encoder(torch.randn(2, 1, 128, 320))
    out2 = encoder(torch.randn(2, 1, 128, 640))
    assert out1.hidden_states.shape == (2, 40, 32)
    assert out2.hidden_states.shape == (2, 80, 32)
    assert out2.attention_mask.shape == (2, 80)


def test_resnet_encoder_pad_mask() -> None:
    encoder = ResNetEncoder(d_model=32, encoder_spec=ENCODER_REGISTRY["resnet"])
    encoder.eval()
    pixel = torch.zeros(1, 1, 128, 320)
    pixel[..., 160:] = 1.0
    out = encoder(pixel)
    assert out.hidden_states.shape == (1, 40, 32)
    assert out.attention_mask.shape == (1, 40)
    # CNN feature width must equal mask width — guards against the AvgPool
    # kernel/stride drifting away from the conv stack.
    assert out.attention_mask.shape[-1] == out.hidden_states.shape[1]
    # Far-right cells of the mask sit fully inside the white-padded region.
    assert out.attention_mask[..., 25:].sum().item() == 0
    # The interior of the content region (away from boundary effects on the
    # very first column from AvgPool zero-padding) is fully unmasked.
    assert out.attention_mask[..., 1:15].sum().item() == 14


def test_decoder_forward_logits_shape() -> None:
    vb = build_default_vocabs()
    decoder = MultiHeadDecoder(vb, _TINY_DECODER_CFG)
    decoder.eval()
    B, L, S = 2, 8, 16
    ids = _make_ids(vb, B, L)
    mask = torch.ones(B, L, dtype=torch.long)
    enc_hidden = torch.randn(B, S, _TINY_DECODER_CFG.d_model)
    out = decoder(ids, mask, enc_hidden, None)
    # Decoder prepends BOS internally, so logits are full length L.
    assert out["logits"]["type"].shape == (B, L, len(vb.type))
    assert out["logits"]["pitch"].shape == (B, L, len(vb.pitch))
    assert out["logits"]["rhythm"].shape == (B, L, len(vb.rhythm))
    assert out["logits"]["attribute"].shape == (B, L, len(vb.attribute))
    # Vocab sizes match the data-side single source of truth.
    assert len(vb.type) == 10
    assert len(vb.pitch) == 180
    assert len(vb.rhythm) == 17
    assert len(vb.attribute) == 31


def test_decoder_position_embedding_is_added() -> None:
    vb = build_default_vocabs()
    torch.manual_seed(0)
    decoder = MultiHeadDecoder(vb, _TINY_DECODER_CFG)
    decoder.eval()
    B, L, S = 2, 6, 8
    ids = _make_ids(vb, B, L)
    mask = torch.ones(B, L, dtype=torch.long)
    enc_hidden = torch.randn(B, S, _TINY_DECODER_CFG.d_model)

    with_pos = decoder(ids, mask, enc_hidden, None)["logits"]["type"].clone()

    # Zero out the positional embedding table; everything else is unchanged.
    orig_pos = decoder.bart.embed_positions.weight.data.clone()
    decoder.bart.embed_positions.weight.data.zero_()
    try:
        no_pos = decoder(ids, mask, enc_hidden, None)["logits"]["type"]
    finally:
        decoder.bart.embed_positions.weight.data.copy_(orig_pos)

    # Logits must change when positions are zeroed — proves embed_positions
    # actually contributes to the forward pass.
    assert not torch.allclose(with_pos, no_pos, atol=1e-5)


def _fake_logits(B: int, L: int, V: int) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randn(B, L, V)


def test_loss_pad_ignored() -> None:
    vb = build_default_vocabs()
    cfg = _TINY_DECODER_CFG
    B, L = 2, 6
    # Logits are full length L (decoder prepends BOS internally).
    logits = {
        "type": _fake_logits(B, L, len(vb.type)),
        "pitch": _fake_logits(B, L, len(vb.pitch)),
        "rhythm": _fake_logits(B, L, len(vb.rhythm)),
        "attribute": _fake_logits(B, L, len(vb.attribute)),
    }
    # Labels with PAD in the final two positions of every stream.
    PAD = Vocabulary.PAD_ID
    full = torch.tensor(
        [
            [5, 6, 7, 2, PAD, PAD],
            [6, 5, 7, 2, PAD, PAD],
        ],
        dtype=torch.long,
    )
    truncated = torch.tensor(
        [
            [5, 6, 7, 2],
            [6, 5, 7, 2],
        ],
        dtype=torch.long,
    )
    labels_full = {n: full.clone() for n in ("type", "pitch", "rhythm", "attribute")}
    labels_trunc = {
        n: truncated.clone() for n in ("type", "pitch", "rhythm", "attribute")
    }
    logits_trunc = {n: logits[n][:, : truncated.size(1), :] for n in logits}

    total_full, _ = compute_loss(logits, labels_full, cfg)
    total_trunc, _ = compute_loss(logits_trunc, labels_trunc, cfg)
    # PAD positions don't contribute, so the two losses are identical.
    assert torch.allclose(total_full, total_trunc, atol=1e-6)


def test_loss_null_masking_flag() -> None:
    vb = build_default_vocabs()
    B, L = 2, 5
    logits = {
        "type": _fake_logits(B, L, len(vb.type)),
        "pitch": _fake_logits(B, L, len(vb.pitch)),
        "rhythm": _fake_logits(B, L, len(vb.rhythm)),
        "attribute": _fake_logits(B, L, len(vb.attribute)),
    }
    NULL = Vocabulary.NULL_ID
    labels = {
        "type": torch.tensor([[1, 5, 6, 7, 8], [1, 5, 6, 7, 8]], dtype=torch.long),
        "pitch": torch.tensor(
            [[1, NULL, NULL, NULL, 2], [1, NULL, NULL, NULL, 2]], dtype=torch.long
        ),
        "rhythm": torch.tensor([[1, 5, 6, 7, 2], [1, 5, 6, 7, 2]], dtype=torch.long),
        "attribute": torch.tensor([[1, 5, 6, 7, 2], [1, 5, 6, 7, 2]], dtype=torch.long),
    }
    cfg_off = ModelConfig(
        d_model=32,
        decoder_layers=2,
        decoder_heads=2,
        decoder_ffn_dim=64,
        max_decoder_positions=64,
        mask_null_in_loss=MaskNullInLoss(pitch=False),
    )
    cfg_on = ModelConfig(
        d_model=32,
        decoder_layers=2,
        decoder_heads=2,
        decoder_ffn_dim=64,
        max_decoder_positions=64,
        mask_null_in_loss=MaskNullInLoss(pitch=True),
    )
    _, per_off = compute_loss(logits, labels, cfg_off)
    _, per_on = compute_loss(logits, labels, cfg_on)
    # Toggling the flag must change the pitch loss (NULL positions are now
    # ignored) but leave the other three heads unchanged.
    assert not torch.allclose(per_off["pitch"], per_on["pitch"])
    assert torch.allclose(per_off["type"], per_on["type"])
    assert torch.allclose(per_off["rhythm"], per_on["rhythm"])
    assert torch.allclose(per_off["attribute"], per_on["attribute"])


def test_loss_eos_weight() -> None:
    vb = build_default_vocabs()
    B, L = 2, 5
    logits = {
        "type": _fake_logits(B, L, len(vb.type)),
        "pitch": _fake_logits(B, L, len(vb.pitch)),
        "rhythm": _fake_logits(B, L, len(vb.rhythm)),
        "attribute": _fake_logits(B, L, len(vb.attribute)),
    }
    EOS = Vocabulary.EOS_ID
    # Each sequence ends in EOS; labels are the targets directly.
    labels = {
        n: torch.tensor([[5, 6, 7, 8, EOS], [6, 5, 7, 8, EOS]], dtype=torch.long)
        for n in ("type", "pitch", "rhythm", "attribute")
    }
    cfg_off = ModelConfig(
        d_model=32,
        decoder_layers=2,
        decoder_heads=2,
        decoder_ffn_dim=64,
        max_decoder_positions=64,
        eos_weight=1.0,
    )
    cfg_on = ModelConfig(
        d_model=32,
        decoder_layers=2,
        decoder_heads=2,
        decoder_ffn_dim=64,
        max_decoder_positions=64,
        eos_weight=10.0,
    )
    _, per_off = compute_loss(logits, labels, cfg_off)
    _, per_on = compute_loss(logits, labels, cfg_on)
    # Up-weighting EOS changes only the type-head loss.
    assert not torch.allclose(per_off["type"], per_on["type"])
    assert torch.allclose(per_off["pitch"], per_on["pitch"])
    assert torch.allclose(per_off["rhythm"], per_on["rhythm"])
    assert torch.allclose(per_off["attribute"], per_on["attribute"])


def _build_tiny_model() -> OMRModel:
    torch.manual_seed(0)
    vb = build_default_vocabs()
    encoder = ResNetEncoder(d_model=32, encoder_spec=ENCODER_REGISTRY["resnet"])
    model = OMRModel(encoder=encoder, vocabs=vb, cfg=_TINY_DECODER_CFG)
    model.eval()
    return model


def test_generate_shape() -> None:
    model = _build_tiny_model()
    pixel = torch.randn(2, 1, 128, 320)
    out = model.generate(pixel, max_length=16)
    for ids in (out.type_ids, out.pitch_ids, out.rhythm_ids, out.attribute_ids):
        assert ids.shape[0] == 2
        assert 1 <= ids.shape[1] <= 16
    assert out.lengths.shape == (2,)
    assert (out.lengths >= 1).all()
    assert (out.lengths <= 16).all()


def test_generate_stops_on_type_eos() -> None:
    model = _build_tiny_model()
    # Force the type head to emit EOS at every step.
    with torch.no_grad():
        head = model.decoder.heads["head_type"]
        head.weight.zero_()
        head.bias.fill_(-1e9)
        head.bias[Vocabulary.EOS_ID] = 1e9
    pixel = torch.randn(1, 1, 128, 320)
    out = model.generate(pixel, max_length=8)
    # After BOS (step 0), step 1's forced EOS finishes the row → lengths = 2.
    assert out.lengths.item() == 2


def test_generate_kv_cache_matches_no_cache() -> None:
    vb = build_default_vocabs()
    torch.manual_seed(0)
    decoder = MultiHeadDecoder(vb, _TINY_DECODER_CFG)
    decoder.eval()
    enc_hidden = torch.randn(1, 12, _TINY_DECODER_CFG.d_model)
    streams = ("type", "pitch", "rhythm", "attribute")

    # `step` takes literal decoder-input tokens (no internal shift). The
    # generation loop feeds [BOS, t0, t1, t2]; replicate that exact sequence.
    seq = [Vocabulary.BOS_ID, 5, 6, 7]

    # No-cache baseline: one step() call over the whole sequence (past=None).
    ids_full = {n: torch.tensor([seq], dtype=torch.long) for n in streams}
    no_cache, _ = decoder.step(ids_full, enc_hidden, None, None)

    # Cached rollout: feed the same tokens one at a time, threading the cache.
    past = None
    chunks: list[torch.Tensor] = []
    for tok in seq:
        feed = {n: torch.tensor([[tok]], dtype=torch.long) for n in streams}
        logits, past = decoder.step(feed, enc_hidden, None, past)
        chunks.append(logits["type"])
    cached = torch.cat(chunks, dim=1)

    assert no_cache["type"].shape == cached.shape == (1, 4, len(vb.type))
    assert torch.allclose(no_cache["type"], cached, atol=1e-5)


def test_integration_forward_loss_backward_postproc() -> None:
    import math

    from src.model.evaluation import run_validation

    vb = build_default_vocabs()
    torch.manual_seed(0)
    encoder = ResNetEncoder(d_model=32, encoder_spec=ENCODER_REGISTRY["resnet"])
    model = OMRModel(encoder=encoder, vocabs=vb, cfg=_TINY_DECODER_CFG)
    model.train()

    B, L = 2, 6
    pixel = torch.randn(B, 1, 128, 320)
    EOS, PAD = Vocabulary.EOS_ID, Vocabulary.PAD_ID
    # Realistic data layout: [content..., EOS, PAD...] with NO leading BOS.
    batch = {
        "pixel_values": pixel,
        "type_ids": torch.tensor(
            [[5, 6, 7, 8, EOS, PAD], [6, 7, 8, EOS, PAD, PAD]], dtype=torch.long
        ),
        "pitch_ids": torch.tensor(
            [[5, 6, 7, 8, EOS, PAD], [6, 7, 8, EOS, PAD, PAD]], dtype=torch.long
        ),
        "rhythm_ids": torch.tensor(
            [[5, 6, 7, 8, EOS, PAD], [6, 7, 8, EOS, PAD, PAD]], dtype=torch.long
        ),
        "attribute_ids": torch.tensor(
            [[5, 6, 7, 8, EOS, PAD], [6, 7, 8, EOS, PAD, PAD]], dtype=torch.long
        ),
        "decoder_attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.long
        ),
        "label_lengths": torch.tensor([5, 4], dtype=torch.long),
    }

    fwd = model(
        pixel_values=batch["pixel_values"],
        type_ids=batch["type_ids"],
        pitch_ids=batch["pitch_ids"],
        rhythm_ids=batch["rhythm_ids"],
        attribute_ids=batch["attribute_ids"],
        decoder_attention_mask=batch["decoder_attention_mask"],
    )
    labels = {
        "type": batch["type_ids"],
        "pitch": batch["pitch_ids"],
        "rhythm": batch["rhythm_ids"],
        "attribute": batch["attribute_ids"],
    }
    total, _ = compute_loss(fwd["logits"], labels, _TINY_DECODER_CFG)
    total.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )

    # End-to-end: generate → IdSeqs → evaluate_batch → finite SER.
    metrics = run_validation(model, [batch], vb, max_length=L + 2)
    assert math.isfinite(metrics.ser)
