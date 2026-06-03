"""LTXAV guide-cond fallback used when the running ComfyUI core lacks PR #13325.

The handler in context_windows.py defers to ``model.resize_cond_for_context_window`` /
``model.map_context_window_to_modalities`` first; these functions are only used when the base
model does not provide them. They are ported from PR #13325's LTXAV model methods so the pack
behaves identically on stock and PR-merged cores, without editing or monkey-patching core.
"""
from __future__ import annotations
import torch
import comfy.model_base
import comfy.ldm.lightricks.symmetric_patchifier


# Sentinel returned by resize_ltxav_cond_for_context_window to mean "omit this cond key for the
# window". Used for keyframe_idxs on guide-less windows so the unmodified model forward takes its
# keyframe_idxs-is-None path, avoiding the empty-tensor branch that PR #13325 guards in core.
DROP = object()


def _is_ltxav(model) -> bool:
    cls = getattr(comfy.model_base, "LTXAV", None)
    return cls is not None and isinstance(model, cls)


def map_context_window_to_modalities(primary_indices, latent_shapes, dim):
    result = [primary_indices]
    if len(latent_shapes) < 2:
        return result

    video_total = latent_shapes[0][dim]

    for i in range(1, len(latent_shapes)):
        mod_total = latent_shapes[i][dim]
        # Map each primary index to its proportional range of modality indices and
        # concatenate in order. Preserves wrapped/strided geometry so the modality
        # attends to the same temporal regions as the primary window.
        mod_indices = []
        seen = set()
        for v_idx in primary_indices:
            a_start = min(int(round(v_idx * mod_total / video_total)), mod_total - 1)
            a_end = min(int(round((v_idx + 1) * mod_total / video_total)), mod_total)
            if a_end <= a_start:
                a_end = a_start + 1
            for a in range(a_start, a_end):
                if a not in seen:
                    seen.add(a)
                    mod_indices.append(a)
        result.append(mod_indices)

    return result


def resize_ltxav_cond_for_context_window(model, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
    if not _is_ltxav(model):
        return None

    # Audio denoise mask — slice using audio modality window
    if cond_key == "audio_denoise_mask" and hasattr(window, 'modality_windows') and window.modality_windows:
        audio_window = window.modality_windows.get(1)
        if audio_window is not None and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
            sliced = audio_window.get_tensor(cond_value.cond, device, dim=2)
            return cond_value._copy_with(sliced)

    # Video denoise mask — split into video + guide portions, slice each
    if cond_key == "denoise_mask" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
        cond_tensor = cond_value.cond
        guide_count = cond_tensor.size(window.dim) - x_in.size(window.dim)
        if guide_count > 0:
            T_video = x_in.size(window.dim)
            video_mask = cond_tensor.narrow(window.dim, 0, T_video)
            guide_mask = cond_tensor.narrow(window.dim, T_video, guide_count)
            sliced_video = window.get_tensor(video_mask, device, retain_index_list=retain_index_list)
            suffix_indices = window.guide_frames_indices
            if suffix_indices:
                idx = tuple([slice(None)] * window.dim + [suffix_indices])
                sliced_guide = guide_mask[idx].to(device)
                return cond_value._copy_with(torch.cat([sliced_video, sliced_guide], dim=window.dim))
            else:
                return cond_value._copy_with(sliced_video)

    # Keyframe indices — regenerate pixel coords for window, select guide positions
    if cond_key == "keyframe_idxs":
        kf_local_pos = window.guide_kf_local_positions
        if not kf_local_pos:
            return DROP  # no guide overlap: drop the key so the model takes its keyframe_idxs-is-None path
        H, W = x_in.shape[3], x_in.shape[4]
        window_len = len(window.index_list)
        # account for causal_window_fix anchor in coord space size
        anchor_idx = getattr(window, 'causal_anchor_index', None)
        if anchor_idx is not None and anchor_idx >= 0:
            window_len += 1
        patchifier = model.diffusion_model.patchifier
        latent_coords = patchifier.get_latent_coords(window_len, H, W, 1, cond_value.cond.device)
        scale_factors = model.diffusion_model.vae_scale_factors
        pixel_coords = comfy.ldm.lightricks.symmetric_patchifier.latent_to_pixel_coords(
            latent_coords,
            scale_factors,
            causal_fix=model.diffusion_model.causal_temporal_positioning)
        tokens = []
        for pos in kf_local_pos:
            tokens.extend(range(pos * H * W, (pos + 1) * H * W))
        pixel_coords = pixel_coords[:, :, tokens, :]

        # Adjust spatial end positions for dilated (downscaled) guides.
        # Each guide entry may have a different downscale factor; expand the
        # per-entry factor to cover all tokens belonging to that entry.
        downscale_factors = window.guide_downscale_factors
        overlap_info = window.guide_overlap_info
        if downscale_factors:
            per_token_factor = []
            for (entry_idx, overlap_count), dsf in zip(overlap_info, downscale_factors):
                per_token_factor.extend([dsf] * (overlap_count * H * W))
            factor_tensor = torch.tensor(per_token_factor, device=pixel_coords.device, dtype=pixel_coords.dtype)
            spatial_end_offset = (factor_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1) - 1) * torch.tensor(
                scale_factors[1:], device=pixel_coords.device, dtype=pixel_coords.dtype,
            ).view(1, -1, 1, 1)
            pixel_coords[:, 1:, :, 1:] += spatial_end_offset

        B = cond_value.cond.shape[0]
        if B > 1:
            pixel_coords = pixel_coords.expand(B, -1, -1, -1)
        return cond_value._copy_with(pixel_coords)

    # Guide attention entries — adjust per-guide counts based on window overlap
    if cond_key == "guide_attention_entries":
        overlap_info = window.guide_overlap_info
        H, W = x_in.shape[3], x_in.shape[4]
        new_entries = []
        for entry_idx, overlap_count in overlap_info:
            e = cond_value.cond[entry_idx]
            new_entries.append({**e,
                "pre_filter_count": overlap_count * H * W,
                "latent_shape": [overlap_count, H, W]})
        return cond_value._copy_with(new_entries)

    return None
