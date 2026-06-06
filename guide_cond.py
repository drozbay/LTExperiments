"""Conditioning-carried guides for LTXV / LTXAV (proof of concept).

Unlike core's LTXVAddGuide, the guide latent is NOT appended to the working latent. It rides on
the conditioning and is injected as extra tokens *inside* ``LTXBaseModel._forward`` (with rope +
``(1 - strength) * sigma`` per-token timestep), then stripped back off before the output is
unpatchified. So there is nothing to crop afterward and the working latent is never touched.

Reaching mid-``_forward`` (between ``_process_input`` and ``_prepare_timestep``) is not possible
through any official wrapper hook, so this installs a **gated, idempotent monkeypatch** of
``LTXBaseModel._forward`` plus the ``LTXV`` / ``LTXAV`` ``extra_conds``. The patches are a complete
no-op unless the ``guide_cond_latents`` conditioning key is present, so normal LTXV/LTXAV sampling
is byte-identical to stock core. This deliberately departs from the rest of the pack's
"no monkeypatch core" approach because the official hooks cannot express this injection; see the
node docstring for the rationale.

LTXV uses single tensors throughout. LTXAV carries video/audio as ``[video, audio]`` lists; guides
are video-only, so injection targets the video slot (index 0) and leaves audio untouched.
"""
from __future__ import annotations
import torch
import node_helpers
import comfy.conds
import comfy.model_base
import comfy.ldm.lightricks.model as ltx_model
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords

GUIDE_LATENTS_KEY = "guide_cond_latents"
GUIDE_COORDS_KEY = "guide_cond_keyframe_idxs"
GUIDE_STRENGTH_KEY = "guide_cond_strength"

# Standalone patchifier for node-side coord computation (matches LTXVModel.patchifier config).
_PATCHIFIER = SymmetricPatchifier(1, start_end=True)

_PATCHED = False


# --------------------------------------------------------------------------------------------
# Node-side helpers (model-free): coordinates + conditioning accumulation
# --------------------------------------------------------------------------------------------
def compute_guide_coords(guide_latent, frame_idx, scale_factors, causal_fix):
    """Pixel-space keyframe coordinates for a guide latent placed at pixel-frame ``frame_idx``.

    Mirrors LTXVAddGuide.add_keyframe_index (without IC-LoRA dilation): patchify for the base grid,
    then offset the temporal channel by frame_idx. Shape: (B, [t,h,w], token, [start,end]).
    """
    _, latent_coords = _PATCHIFIER.patchify(guide_latent)
    pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=causal_fix)
    pixel_coords[:, 0] += frame_idx
    return pixel_coords


def _existing_coords(cond):
    for t in cond:
        v = t[1].get(GUIDE_COORDS_KEY)
        if v is not None:
            return v
    return None


def append_guide(cond, guide_latent, pixel_coords, strength):
    """Store one guide on the conditioning, accumulating across chained nodes.

    Coords are a tensor (concatenated along the token dim; node_helpers append=True would add
    tensors elementwise, which is wrong). Latents/strengths are lists (append=True concatenates),
    kept aligned with the coord token order.
    """
    existing = _existing_coords(cond)
    if existing is not None:
        pixel_coords = torch.cat([existing, pixel_coords], dim=2)
    cond = node_helpers.conditioning_set_values(cond, {GUIDE_COORDS_KEY: pixel_coords})
    cond = node_helpers.conditioning_set_values(
        cond, {GUIDE_LATENTS_KEY: [guide_latent], GUIDE_STRENGTH_KEY: [strength]}, append=True
    )
    return cond


# --------------------------------------------------------------------------------------------
# Forward-injection helpers (model-side), ported from the core branch; list-aware for LTXAV
# --------------------------------------------------------------------------------------------
def _inject(self, x, pixel_coords, timestep, batch_size, merged_args):
    guide_latents = merged_args[GUIDE_LATENTS_KEY]
    guide_coords = merged_args[GUIDE_COORDS_KEY]
    guide_strength = merged_args.get(GUIDE_STRENGTH_KEY, None)

    is_av = isinstance(x, list)
    vx = x[0] if is_av else x
    v_coords = pixel_coords[0] if is_av else pixel_coords

    main_len = vx.shape[1]

    # Recover per-batch sigma from the (video) timestep. With no base denoise_mask it is uniform;
    # amax picks the full-noise value if a base mask were ever present.
    sigma = timestep.reshape(batch_size, -1).amax(dim=1, keepdim=True)  # [B, 1]

    timestep_parts = [sigma.expand(batch_size, main_len)]
    guide_tokens = []
    for i, lat in enumerate(guide_latents):
        gx, _ = self.patchifier.patchify(lat.to(device=vx.device, dtype=vx.dtype))
        gx = self.patchify_proj(gx)  # same input projection (in_channels -> inner_dim) base tokens get
        guide_tokens.append(gx)
        s = guide_strength[i] if guide_strength is not None else 1.0
        timestep_parts.append(sigma.expand(batch_size, gx.shape[1]) * max(0.0, 1.0 - float(s)))

    vx = torch.cat([vx] + guide_tokens, dim=1)
    v_coords = torch.cat([v_coords, guide_coords.to(v_coords.device)], dim=2)
    timestep = torch.cat(timestep_parts, dim=1)

    if is_av:
        x = [vx, x[1]]
        pixel_coords = [v_coords, pixel_coords[1]]
    else:
        x, pixel_coords = vx, v_coords
    return x, pixel_coords, timestep, main_len


def _strip(x, embedded_timestep, main_len):
    if isinstance(x, list):
        x[0] = x[0][:, :main_len]
        v_emb = embedded_timestep[0]
        if not torch.is_tensor(v_emb):  # CompressedTimestep (per-frame) -> expand to per-token
            v_emb = v_emb.expand()
        embedded_timestep[0] = v_emb[:, :main_len]
    else:
        x = x[:, :main_len]
        embedded_timestep = embedded_timestep[:, :main_len]
    return x, embedded_timestep


# --------------------------------------------------------------------------------------------
# Gated monkeypatch installer
# --------------------------------------------------------------------------------------------
def _make_patched_forward(orig_forward):
    def _forward(self, x, timestep, context, attention_mask, frame_rate=25, transformer_options={},
                 keyframe_idxs=None, denoise_mask=None, **kwargs):
        # Gate: no guides -> delegate to the original forward, byte-identical to stock core.
        if kwargs.get(GUIDE_LATENTS_KEY) is None and transformer_options.get(GUIDE_LATENTS_KEY) is None:
            return orig_forward(self, x, timestep, context, attention_mask, frame_rate,
                                transformer_options, keyframe_idxs, denoise_mask=denoise_mask, **kwargs)

        # Guide path: this mirrors core LTXBaseModel._forward with the two guarded inject/strip calls.
        if isinstance(x, list):
            input_dtype = x[0].dtype
            batch_size = x[0].shape[0]
        else:
            input_dtype = x.dtype
            batch_size = x.shape[0]

        merged_args = {**transformer_options, **kwargs}
        x, pixel_coords, additional_args = self._process_input(x, keyframe_idxs, denoise_mask, **merged_args)
        merged_args.update(additional_args)

        x, pixel_coords, timestep, main_len = _inject(self, x, pixel_coords, timestep, batch_size, merged_args)

        timestep, embedded_timestep, prompt_timestep = self._prepare_timestep(timestep, batch_size, input_dtype, **merged_args)
        merged_args["prompt_timestep"] = prompt_timestep
        context, attention_mask = self._prepare_context(context, batch_size, x, attention_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, input_dtype)
        pe = self._prepare_positional_embeddings(pixel_coords, frame_rate, input_dtype)
        self_attention_mask = self._build_guide_self_attention_mask(x, transformer_options, merged_args)

        x = self._process_transformer_blocks(
            x, context, attention_mask, timestep, pe,
            transformer_options=transformer_options,
            self_attention_mask=self_attention_mask,
            **merged_args,
        )

        x, embedded_timestep = _strip(x, embedded_timestep, main_len)
        # keyframe_idxs is None on this path (guides never entered the latent), so _process_output
        # skips its grid-mask scatter branch.
        x = self._process_output(x, embedded_timestep, None, **merged_args)
        return x

    _forward._ltex_guide_cond = True
    return _forward


def _make_patched_extra_conds(orig_extra_conds):
    def extra_conds(self, **kwargs):
        out = orig_extra_conds(self, **kwargs)
        guide_latents = kwargs.get(GUIDE_LATENTS_KEY, None)
        if guide_latents is not None:
            # process_latent_in here because the guides bypass the main latent (the latent-carried
            # path gets it for free via the sampler). Safe for LTXAV: its latent format is LTXAV(LTXV),
            # so a plain video guide gets the same video normalization as the video stream.
            out[GUIDE_LATENTS_KEY] = comfy.conds.CONDList([self.process_latent_in(l) for l in guide_latents])
        guide_coords = kwargs.get(GUIDE_COORDS_KEY, None)
        if guide_coords is not None:
            out[GUIDE_COORDS_KEY] = comfy.conds.CONDRegular(guide_coords)
        guide_strength = kwargs.get(GUIDE_STRENGTH_KEY, None)
        if guide_strength is not None:
            out[GUIDE_STRENGTH_KEY] = comfy.conds.CONDConstant(guide_strength)
        return out

    extra_conds._ltex_guide_cond = True
    return extra_conds


def install():
    """Install the gated patches once. Idempotent across module reloads (guarded by a marker
    attribute on the patched callables)."""
    global _PATCHED
    if _PATCHED:
        return

    if not getattr(ltx_model.LTXBaseModel._forward, "_ltex_guide_cond", False):
        ltx_model.LTXBaseModel._forward = _make_patched_forward(ltx_model.LTXBaseModel._forward)

    for model_cls in (comfy.model_base.LTXV, getattr(comfy.model_base, "LTXAV", None)):
        if model_cls is None:
            continue
        if not getattr(model_cls.extra_conds, "_ltex_guide_cond", False):
            model_cls.extra_conds = _make_patched_extra_conds(model_cls.extra_conds)

    _PATCHED = True
