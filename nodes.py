"""LTExperiments context-window nodes (V3 io).

Patches the model with this pack's forked context-window handler (see context_windows.py) for
LTXV / LTX-2 sampling.
"""
from comfy_api.latest import ComfyExtension, io
import torch
import nodes
import node_helpers
import comfy.utils
import comfy.model_management

from . import context_windows as cw
from . import perwindow_guides as pw
from . import guide_cond as gc

CATEGORY = "LTExperiments"


def _encode_guide(vae, latent_width, latent_height, images):
    """Encode guide image(s) to latent space sized to match the sampling latent (mirrors LTXVAddGuide.encode)."""
    scale_factors = vae.downscale_index_formula  # (time, height, width)
    time_scale_factor = scale_factors[0]
    images = images[:(images.shape[0] - 1) // time_scale_factor * time_scale_factor + 1]
    target_w = int(latent_width * scale_factors[2])
    target_h = int(latent_height * scale_factors[1])
    pixels = comfy.utils.common_upscale(images.movedim(-1, 1), target_w, target_h, "bilinear", crop="center").movedim(1, -1)
    return vae.encode(pixels[:, :, :, :3])


def _existing_perwindow_guides(cond):
    for t in cond:
        v = t[1].get(pw.PERWINDOW_KEY)
        if v:
            return list(v)
    return []


class LTEx_ContextWindowsManualNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTEx_ContextWindowsManual",
            display_name="LTEx Context Windows (Manual)",
            category=CATEGORY,
            description="Manually set context windows.",
            inputs=[
                io.Model.Input("model", tooltip="The model to apply context windows to during sampling."),
                io.Int.Input("context_length", min=1, default=16, tooltip="The length of the context window."),
                io.Int.Input("context_overlap", min=0, default=4, tooltip="The overlap of the context window."),
                io.Combo.Input("context_schedule", options=[
                    cw.ContextSchedules.STATIC_STANDARD,
                    cw.ContextSchedules.UNIFORM_STANDARD,
                    cw.ContextSchedules.UNIFORM_LOOPED,
                    cw.ContextSchedules.BATCHED,
                    ], default=cw.ContextSchedules.STATIC_STANDARD, tooltip="Step-dependent scheduling algorithm for context windows."),
                io.Int.Input("context_stride", min=1, default=1, tooltip="The stride of the context window; only applicable to uniform schedules."),
                io.Boolean.Input("closed_loop", default=False, tooltip="Whether to close the context window loop; only applicable to looped schedules."),
                io.Combo.Input("fuse_method", options=cw.ContextFuseMethods.LIST_STATIC, default=cw.ContextFuseMethods.PYRAMID, tooltip="The method to use to fuse the context windows."),
                io.Int.Input("dim", min=0, max=5, default=0, tooltip="The dimension to apply the context windows to."),
                io.Boolean.Input("freenoise", default=False, tooltip="Whether to apply FreeNoise noise shuffling, improves window blending."),
                io.String.Input("cond_retain_index_list", default="", tooltip="List of latent indices to retain in the conditioning tensors for each window. For concat-style I2V models (e.g. Wan I2V, HunyuanVideo I2V, Cosmos I2V, SVD) the encoded start image lives in the c_concat conditioning channels; setting this to '0' will retain that start image content at sub-pos 0 of every window."),
                io.Boolean.Input("split_conds_to_windows", default=False, tooltip="Whether to split multiple conditionings (created by ConditionCombine) to each window based on region index."),
                io.String.Input("latent_retain_index_list", default="", tooltip="List of latent indices to retain in the noise latent itself for each window. Use for workflows where reference content (e.g. a start image) lives directly in the noise latent rather than in separate conditioning channels (e.g. inplace-style I2V like LTXV, AnimateDiff). Independent of cond_retain_index_list."),
                io.Boolean.Input("causal_window_fix", default=True, tooltip="Whether to add a causal fix frame to non-0-indexed context windows."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with context windows applied during sampling."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: io.Model.Type, context_length: int, context_overlap: int, context_schedule: str, context_stride: int, closed_loop: bool, fuse_method: str, dim: int, freenoise: bool,
                cond_retain_index_list: list[int]=[], split_conds_to_windows: bool=False, latent_retain_index_list: list[int]=[], causal_window_fix: bool=True) -> io.Model:
        model = model.clone()
        model.model_options["context_handler"] = cw.IndexListContextHandler(
            context_schedule=cw.get_matching_context_schedule(context_schedule),
            fuse_method=cw.get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=dim,
            freenoise=freenoise,
            cond_retain_index_list=cond_retain_index_list,
            split_conds_to_windows=split_conds_to_windows,
            latent_retain_index_list=latent_retain_index_list,
            causal_window_fix=causal_window_fix,
        )
        # make memory usage calculation only take into account the context window latents
        cw.create_prepare_sampling_wrapper(model)
        if freenoise: # no other use for this wrapper at this time
            cw.create_sampler_sample_wrapper(model)
        return io.NodeOutput(model)


class LTEx_LTXVContextWindowsNode(LTEx_ContextWindowsManualNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "LTEx_LTXVContextWindows"
        schema.display_name = "LTEx LTXV Context Windows"
        schema.description = "Set context windows for LTXV / LTX-2 models."
        schema.inputs = [
            io.Model.Input("model", tooltip="The model to apply context windows to during sampling."),
            io.Int.Input("context_length", min=1, max=nodes.MAX_RESOLUTION, step=8, default=145, tooltip="The length of the context window in real frames. Must be 8*n + 1."),
            io.Int.Input("context_overlap", min=0, step=8, default=40, tooltip="The overlap of the context window in real frames."),
            io.Combo.Input("context_schedule", options=[
                cw.ContextSchedules.STATIC_STANDARD,
                cw.ContextSchedules.UNIFORM_STANDARD,
                cw.ContextSchedules.UNIFORM_LOOPED,
                cw.ContextSchedules.BATCHED,
                ], default=cw.ContextSchedules.UNIFORM_STANDARD, tooltip="Step-dependent scheduling algorithm for context windows."),
            io.Int.Input("context_stride", min=1, default=1, tooltip="The stride of the context window; only applicable to uniform schedules.", advanced=True),
            io.Boolean.Input("closed_loop", default=False, tooltip="Whether to close the context window loop; only applicable to looped schedules.", advanced=True),
            io.Combo.Input("fuse_method", options=cw.ContextFuseMethods.LIST_STATIC, default=cw.ContextFuseMethods.PYRAMID, tooltip="The method to use to fuse the context windows."),
            io.Boolean.Input("freenoise", default=True, tooltip="Whether to apply FreeNoise noise shuffling, improves window blending.", advanced=True),
            io.Boolean.Input("retain_first_frame", default=False, tooltip="Retain the first latent frame in every context window (may help retain initial reference)."),
            io.Boolean.Input("split_conds_to_windows", default=False, tooltip="Whether to split multiple conditionings (created by ConditionCombine) to each window based on region index.", advanced=True),
        ]
        return schema

    @classmethod
    def execute(cls, model: io.Model.Type, context_length: int, context_overlap: int, context_schedule: str, fuse_method: str, freenoise: bool,
                retain_first_frame: bool=False, split_conds_to_windows: bool=False, context_stride: int=1, closed_loop: bool=False) -> io.Model:
        context_length = max(((context_length - 1) // 8) + 1, 1)  # at least length 1
        context_overlap = max(context_overlap // 8, 0)  # at least overlap 0
        retain_index_list = "0" if retain_first_frame else ""
        return super().execute(model, context_length, context_overlap, context_schedule, context_stride, closed_loop, fuse_method, dim=2, freenoise=freenoise,
                               cond_retain_index_list=retain_index_list, latent_retain_index_list=retain_index_list, split_conds_to_windows=split_conds_to_windows)


class LTEx_LTXVAddPerWindowGuideNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTEx_LTXVAddPerWindowGuide",
            display_name="LTEx LTXV Add Per-Window Guide",
            category=CATEGORY,
            description="Inject a guide image at the same relative position in every context window. Requires the LTEx LTXV Context Windows node on the model.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae", tooltip="VAE used to encode the guide image."),
                io.Latent.Input("latent", tooltip="The latent being sampled. Used to size the guide encode; not modified."),
                io.Image.Input("image", tooltip="Guide image. Multiple frames fill consecutive positions starting at index."),
                io.Int.Input("rel_index", default=-1, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION,
                             tooltip="Position of the guide within every window, counted in latent frames. 0 is the window's first frame; use negative values to place it before the first frame."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="How strongly the guide influences each window. Accepts a single value, or a list of per-window strengths (e.g. from a String to Float List node) indexed by window order, clamping to the last value; set a window's value to 0 to skip the guide for that window."),
                io.Mask.Input("attention_mask", optional=True, tooltip="Optional mask limiting where the guide applies."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, image, rel_index: int, strength, attention_mask=None) -> io.NodeOutput:
        _, _, _, latent_height, latent_width = latent["samples"].shape
        t = _encode_guide(vae, latent_width, latent_height, image).to(comfy.model_management.intermediate_device())
        guide = {
            "latent": t,
            "rel_index": int(rel_index),
            "strength": [float(x) for x in strength] if isinstance(strength, (list, tuple)) else float(strength),
            "pixel_mask": attention_mask.unsqueeze(0).unsqueeze(0) if attention_mask is not None else None,
        }
        outs = []
        for cond in (positive, negative):
            guides = [*_existing_perwindow_guides(cond), guide]
            outs.append(node_helpers.conditioning_set_values(cond, {pw.PERWINDOW_KEY: guides}))
        return io.NodeOutput(outs[0], outs[1])


class LTEx_LTXVAddGuideConditioningNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        from comfy_extras.nodes_lt import ICLoRAParameters  # lazy: dodge custom-node load-order issues
        return io.Schema(
            node_id="LTEx_LTXVAddGuideConditioning",
            display_name="LTEx LTXVAddGuide (Cond Only)",
            category=CATEGORY,
            description="Proof of concept: carry an LTXV / LTX-2 guide on the conditioning instead of "
                        "appending it to the working latent (the LTXVAddGuide approach). The guide is "
                        "injected as extra tokens inside the model forward and stripped before output, "
                        "so no LTXVCropGuides node is needed and the latent is never modified. Works for "
                        "LTXV and LTXAV (video stream). Installs a gated monkeypatch of the LTX forward "
                        "that is a no-op unless this node is used. Drop-in alternative to LTXVAddGuide "
                        "for A/B testing the conditioning-carried path.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent", tooltip="The latent being sampled. Used only to size the guide encode; not modified."),
                io.Image.Input("image", tooltip="Guide image or video. Must be 8*n + 1 frames or it is cropped to the nearest."),
                io.Int.Input("frame_idx", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION,
                             tooltip="Pixel-frame index to place the guide at. For 9+ frame guides, frame_idx must be divisible by 8 (rounded down). Negative values count from the end."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="How strongly the guide pins its frames. 1.0 locks them (clean), lower loosens; 0 ignores the guide."),
                io.Mask.Input("attention_mask", optional=True,
                              tooltip="Optional pixel-space spatial mask. Controls per-region guide influence via "
                                      "self-attention, multiplied by strength. Same semantics as LTXVAddGuide."),
                ICLoRAParameters.Input("iclora_parameters", optional=True,
                                       tooltip="Optional IC-LoRA parameters (e.g. reference_downscale_factor > 1) from a "
                                               "Get IC-LoRA Parameters node. The low-res reference is spread across the "
                                               "full-res grid via dilated RoPE coordinates; the latent is still untouched."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent_passthrough", tooltip="Input latent, returned unmodified"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, image, frame_idx: int, strength: float, attention_mask=None, iclora_parameters=None) -> io.NodeOutput:
        from comfy_extras.nodes_lt import LTXVAddGuide, _append_guide_attention_entry  # reuse core's encode + frame-index snapping + attn entries

        gc.install()  # idempotent; gated monkeypatch is a no-op unless guide keys are present

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        _, _, latent_length, latent_height, latent_width = latent_image.shape

        # IC-LoRA references encode at 1/N spatial resolution and are spread across the full-res grid.
        latent_downscale_factor = LTXVAddGuide.get_reference_downscale_factor(iclora_parameters)
        if latent_downscale_factor > 1 and (latent_width % latent_downscale_factor or latent_height % latent_downscale_factor):
            raise ValueError(
                f"Latent spatial size {latent_width}x{latent_height} must be divisible by "
                f"reference_downscale_factor {latent_downscale_factor} from the IC-LoRA parameters."
            )

        # Mirror LTXVAddGuide's causal first-frame handling for mid-video multi-frame guides.
        time_scale_factor = scale_factors[0]
        num_frames_to_keep = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        resolved_frame_idx = frame_idx
        if frame_idx < 0:
            resolved_frame_idx = max((latent_length - 1) * time_scale_factor + 1 + frame_idx, 0)
        causal_fix = resolved_frame_idx == 0 or num_frames_to_keep == 1

        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)

        image, t = LTXVAddGuide.encode(vae, latent_width, latent_height, image, scale_factors, latent_downscale_factor)

        if not causal_fix:
            t = t[:, :, 1:, :, :]
            image = image[1:]

        # Snap/resolve frame_idx (num_keyframes is 0 here since nothing is added to the latent).
        frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
            positive, latent_length, len(image), frame_idx, scale_factors, latent_shape=latent_image.shape
        )
        if latent_idx + t.shape[2] > latent_length:
            raise ValueError("Guide frames exceed the length of the latent sequence.")

        t = t.to(comfy.model_management.intermediate_device())
        coords = gc.compute_guide_coords(t, frame_idx, scale_factors, causal_fix, latent_downscale_factor)

        positive = gc.append_guide(positive, t, coords, float(strength))
        negative = gc.append_guide(negative, t, coords, float(strength))

        # Per-guide attention control: reuse core's entry helper. No latent dilation here, so
        # pre_filter_count is just the guide's token count and latent_shape is its [F, H, W].
        pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
        positive, negative = _append_guide_attention_entry(
            positive, negative, pre_filter_count, list(t.shape[2:]), strength=strength, attention_mask=attention_mask,
        )
        return io.NodeOutput(positive, negative, latent)


class LTExperimentsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTEx_LTXVContextWindowsNode,
            LTEx_LTXVAddPerWindowGuideNode,
            LTEx_LTXVAddGuideConditioningNode,
        ]


def comfy_entrypoint():
    return LTExperimentsExtension()
