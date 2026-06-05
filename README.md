# LTExperiments

Context-window sampling for LTXV and LTX-2 (LTXAV) video models in native ComfyUI, with support for the multimodal video and audio path and for splitting IC-LoRA guides across windows.

This pack is a self-contained extraction of [Comfy-Org/ComfyUI#13325](https://github.com/Comfy-Org/ComfyUI/pull/13325). It uses ComfyUI's existing extension points, the `model_options["context_handler"]` slot and the sampling wrappers, and it does not edit or monkey-patch core. It works whether or not that pull request has been merged into your copy of ComfyUI. When the base model already provides the guide hooks the pack defers to them, otherwise it uses its own bundled implementation.

## Nodes

### LTEx LTXV Context Windows

Turns on context-window sampling for a model. Place it between your model and the sampler.

- Frame inputs are given in real pixel frames. `context_length` snaps to the nearest `8n+1` and `context_overlap` snaps to a multiple of 8.
- `retain_first_frame` keeps the first latent frame in both the conditioning and the noise latent for every window, which is the right setup for inplace-style LTX-2 first-frame conditioning and AnimateDiff-style image-to-video.
- Advanced settings (stride, closed loop, freenoise) are collapsed by default.

### LTEx LTXV Add Per-Window Guide

Inserts a guide image into every context window at the same position relative to that window. A normal LTXVAddGuide pins a guide to one absolute position in the video, so it only appears in the windows that cover it. This node repeats the guide in each window instead, which makes it a useful anchor for the start, or any chosen point, of every window. The guide rides on the conditioning and never adds frames to the working latent, so the latent stays clean between nodes. It needs the Context Windows node on the same model.

`rel_index` sets where the guide sits inside each window, measured in latent frames:

- `0` places it on the window's first frame.
- A positive value places it that many frames into the window.
- A negative value places it before the first frame, which is a real rope position rather than an index counted from the end.

`strength` accepts a single value or a list of per-window strengths:

- A single value applies to every window.
- A list applies its first value to the first window, its second to the second, and so on, reusing the last value for any further windows. The String to Float List node from ComfyUI-KJNodes is a convenient way to build one.
- A window value of `0` skips the guide for that window.
- The list maps to windows by the standard_static layout, so it stays consistent even when you sample with a uniform schedule.

You can chain several of these nodes. Each one carries its own image, position, and strengths, and they are applied independently within each window.

## IC-LoRA guides

The pack also works with guides produced by the native LTXVAddGuide node, with no extra setup beyond adding your guides as usual. Those guides are stripped from the working latent and then reinserted only into the windows they overlap, with `keyframe_idxs` and `guide_attention_entries` regenerated for each window. Downscaled IC-LoRA guides are supported.

## Requirements

- A native ComfyUI install with LTXV or LTX-2 support.
- No extra Python dependencies.

## Debugging

Set the environment variable `LTEX_DEBUG` to 1 before starting ComfyUI to log the window index, position, and strength for each per-window guide as it is injected. Leave the variable unset for normal use.
