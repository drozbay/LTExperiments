# LTExperiments

Context-window sampling for **LTXV / LTX-2 (LTXAV)** video models in native ComfyUI, including the
multimodal (video + audio) path and **IC-LoRA guide** splitting — guides placed at arbitrary positions
are spliced into each window they overlap.

This is a self-contained extraction of the work in
[Comfy-Org/ComfyUI#13325](https://github.com/Comfy-Org/ComfyUI/pull/13325). It plugs into ComfyUI's
existing extension points (`model_options["context_handler"]` + sampling wrappers) and **does not edit
or monkey-patch core**. It works whether or not that PR is merged into your ComfyUI: when the base
model provides the guide hooks it defers to them, otherwise it uses its own bundled implementation.

## Node

**LTEx LTXV Context Windows** — drop it between your model and the sampler. Inputs are in real (pixel)
frames; `context_length` snaps to `8n+1`, `context_overlap` to multiples of 8. `retain_first_frame`
keeps latent sub-pos 0 in both the conditioning and the noise latent (the right combo for
inplace-style LTX-2 first-frame / AnimateDiff-style I2V). Advanced settings (stride, closed_loop,
freenoise) are collapsed by default.

## IC-LoRA guides

Guide support consumes the conds produced by the native **LTXVAddGuide** node — no extra setup beyond
adding your guide(s) as usual. Guides are stripped from the working latent, then re-injected only into
the windows they overlap, with `keyframe_idxs` / `guide_attention_entries` regenerated per window
(downscaled IC-LoRA guides included).

## Requirements

Native ComfyUI with LTXV / LTX-2 support. No extra Python dependencies.
