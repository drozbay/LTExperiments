"""Per-window guide injection for LTXV context windows.

A per-window guide is a clean guide latent injected into each context window at a fixed
window-relative position, rather than at one absolute position like a normal LTXVAddGuide guide.
Its per-window strength can vary, and a strength of 0 skips a given window. It rides on the
conditioning (key ``PERWINDOW_KEY``) and never touches the working latent. The engine appends it
to a window's token sequence, manufactures the matching ``keyframe_idxs``,
``guide_attention_entries``, and ``denoise_mask`` for that window, then strips it back off the
output. The guide is physically a tail token, and its position is purely its RoPE coordinate, so
``rel_index`` may be negative (a coordinate before the window's first frame).
"""
from __future__ import annotations
import os
import logging
import torch
import comfy.conds
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords

PERWINDOW_KEY = "ltex_perwindow_guides"

LTEX_DEBUG = bool(os.environ.get("LTEX_DEBUG"))  # off by default; set LTEX_DEBUG=1 to trace per-window guide injection


def _dbg(msg):
    if LTEX_DEBUG:
        logging.info("[LTEx perwindow] " + msg)


def get_perwindow_guides(conds) -> list | None:
    """Return the raw per-window guide dicts carried on the conditioning, if any.

    The guides ride as a top-level cond-dict key (set via conditioning_set_values); unrecognized
    keys are not wrapped into model_conds by extra_conds, so they live at the top level here.
    """
    for cond_list in conds:
        if cond_list is None:
            continue
        for cond_dict in cond_list:
            entry = cond_dict.get(PERWINDOW_KEY)
            if entry:
                return entry
    return None


def prepare(model, guides: list) -> list:
    """Move guide latents into model latent space once, caching alongside the metadata.

    Returns a list of dicts: {latent (process_latent_in'd, on intermediate device), rel_index,
    strength, pixel_mask, latent_shape [F,H,W]}.
    """
    prepared = []
    for g in guides:
        latent = model.process_latent_in(g["latent"])
        prepared.append({
            "latent": latent,
            "rel_index": int(g["rel_index"]),
            "strength": g.get("strength", 1.0),  # float, or a per-window list of floats
            "pixel_mask": g.get("pixel_mask"),
            "latent_shape": list(latent.shape[2:]),  # [F, H, W]
        })
    return prepared


def resolve_strength(guide, canonical_index: int) -> float:
    """Resolve a guide's strength for a given canonical (standard_static) window index.

    A scalar applies to every window; a list indexes by canonical window, clamping to the last entry.
    """
    s = guide["strength"]
    if isinstance(s, (list, tuple)):
        if not s:
            return 1.0
        return float(s[min(int(canonical_index), len(s) - 1)])
    return float(s)


def active_guides(prepared: list, canonical_index: int) -> list:
    """Guides whose resolved strength for this window is > 0; the rest are skipped entirely."""
    return [p for p in prepared if resolve_strength(p, canonical_index) > 0.0]


def total_frame_count(prepared: list) -> int:
    return sum(p["latent_shape"][0] for p in prepared)


def window_guide_latent(prepared: list, dim: int, device) -> torch.Tensor:
    """Concatenated guide latents to append to a window's primary-modality slice."""
    frames = [p["latent"].to(device) for p in prepared]
    return torch.cat(frames, dim=dim)


def _guide_pixel_coords(model, guide_latent, rel_index, device):
    """Pixel (RoPE) coords for one guide latent placed at window-local position rel_index.

    Mirrors LTXVAddGuide.add_keyframe_index: patchify for the base grid, then offset the temporal
    channel by rel_index in pixel space (one latent step == time scale factor), bypassing the causal
    clamp so negative positions survive. rel_index is measured from the window's first slot, so a
    causal_window_fix anchor (which occupies that slot) sits between rel_index 0 and -1: the guide at
    -1 lands one step before the anchor, not on it.
    """
    patchifier = model.diffusion_model.patchifier
    scale_factors = model.diffusion_model.vae_scale_factors
    time_scale_factor = scale_factors[0]
    _, latent_coords = patchifier.patchify(guide_latent.to(device))
    # causal_fix only affects the base frame-0 token; the explicit offset below carries position.
    pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=False)
    pixel_coords[:, 0] += rel_index * time_scale_factor
    return pixel_coords


def apply_to_window_cond(new_cond_item: dict, prepared: list, model, window, x_in: torch.Tensor):
    """Extend (or create) keyframe_idxs / guide_attention_entries / denoise_mask for a window so
    the appended per-window guide tokens are treated as clean, RoPE-positioned guide tokens.

    Order is [existing guide tokens][per-window guide tokens], matching the latent append order.
    """
    device = x_in.device  # build every per-window cond on the working-latent device
    dim = window.dim
    H, W = x_in.shape[3], x_in.shape[4]
    anchor_idx = getattr(window, "causal_anchor_index", None)
    anchor_shift = 1 if (anchor_idx is not None and anchor_idx >= 0) else 0
    # window-local video length (frames the model will denoise), excluding any appended guides
    video_len = len(window.index_list) + anchor_shift

    ci = getattr(window, "canonical_index", 0)
    active = active_guides(prepared, ci)
    if not active:
        _dbg(f"window {window.index_list[0]}-{window.index_list[-1]} canonical={ci} | no guide (all strengths 0)")
        return  # every guide is strength 0 for this window: inject nothing, leave conds untouched

    win = f"{window.index_list[0]}-{window.index_list[-1]}"
    coords = []
    entries = []
    strengths = []
    for i, p in enumerate(active):
        s = resolve_strength(p, ci)
        strengths.append(s)
        c = _guide_pixel_coords(model, p["latent"], p["rel_index"], device)
        coords.append(c)
        pixel_mask = p["pixel_mask"]
        entries.append({
            "pre_filter_count": p["latent_shape"][0] * H * W,
            "strength": s,
            "pixel_mask": pixel_mask.to(device) if isinstance(pixel_mask, torch.Tensor) else None,
            "latent_shape": [p["latent_shape"][0], H, W],
        })
        if LTEX_DEBUG:
            _dbg(f"window {win} canonical={ci} | guide#{i} rel_index={p['rel_index']} "
                 f"coord_t={c[0, 0, 0, 0].item():.1f} strength={s:.3f}")
    pw_coords = torch.cat(coords, dim=2)

    # keyframe_idxs: append our coords after any existing (already window-local) guide coords.
    existing_kf = new_cond_item.get("keyframe_idxs")
    if existing_kf is not None and hasattr(existing_kf, "cond") and existing_kf.cond is not None:
        merged = torch.cat([existing_kf.cond.to(device), pw_coords], dim=2)
        new_cond_item["keyframe_idxs"] = existing_kf._copy_with(merged)
    else:
        new_cond_item["keyframe_idxs"] = comfy.conds.CONDRegular(pw_coords)

    # guide_attention_entries: append after any existing entries.
    existing_entries = new_cond_item.get("guide_attention_entries")
    if existing_entries is not None and hasattr(existing_entries, "cond") and existing_entries.cond:
        merged_entries = [*existing_entries.cond, *entries]
        new_cond_item["guide_attention_entries"] = existing_entries._copy_with(merged_entries)
    else:
        new_cond_item["guide_attention_entries"] = comfy.conds.CONDConstant(entries)

    # denoise_mask: append a keep-mask (1 - strength) for our tokens; build a video portion if none.
    B = x_in.shape[0]
    pw_mask_parts = [
        torch.full((B, 1, p["latent_shape"][0], H, W), max(0.0, 1.0 - s),
                   device=device, dtype=x_in.dtype)
        for p, s in zip(active, strengths)
    ]
    pw_mask = torch.cat(pw_mask_parts, dim=dim)
    existing_dm = new_cond_item.get("denoise_mask")
    if existing_dm is not None and hasattr(existing_dm, "cond") and isinstance(existing_dm.cond, torch.Tensor):
        base = existing_dm.cond.to(device)
        if base.shape[3] != H or base.shape[4] != W:
            base = base.expand(-1, -1, -1, H, W)
        merged_mask = torch.cat([base, pw_mask], dim=dim)
        new_cond_item["denoise_mask"] = existing_dm._copy_with(merged_mask)
    else:
        video_mask = torch.ones((B, 1, video_len, H, W), device=device, dtype=x_in.dtype)
        merged_mask = torch.cat([video_mask, pw_mask], dim=dim)
        new_cond_item["denoise_mask"] = comfy.conds.CONDRegular(merged_mask)
