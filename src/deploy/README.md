# `src.deploy` — Integration & Deployment (Owner: Member D)

End-to-end pipeline glue, Streamlit demo UI, and Docker / packaging concerns.
Imports from `src.data`, `src.model`, and `src.postproc` to wire the full
inference path: **image → encoder transform → VED (4-head decoder) →
postproc → Jianpu**.

## Inference flow

1. Take a user-uploaded staff image.
2. Apply the matching `EncoderSpec` eval transform from `src.data.get_encoder_spec(...)`
   to produce a `pixel_values` tensor in the same shape the trained checkpoint
   expects.
3. Run the model; receive four decoupled ID streams
   (`type` / `pitch` / `rhythm` / `attribute`).
4. Hand the streams to `src.postproc` for Jianpu rendering.
5. Display the rendered Jianpu alongside the original image in the Streamlit
   UI.

Keep the encoder name stored next to the checkpoint so step 2 picks the right
transform — otherwise a ViT checkpoint silently mis-handles a ResNet-shaped
batch.
