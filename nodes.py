"""LTExperiments context-window nodes (V3 io).

Patches the model with this pack's forked context-window handler (see context_windows.py) for
LTXV / LTX-2 sampling.
"""
from comfy_api.latest import ComfyExtension, io
import nodes

from . import context_windows as cw

CATEGORY = "LTExperiments"


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


class LTExperimentsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTEx_LTXVContextWindowsNode,
        ]


def comfy_entrypoint():
    return LTExperimentsExtension()
