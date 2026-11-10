"""One repair job, end to end, independent of how ComfyUI is reached.

``run_job`` takes validated input and a ``comfy_runner`` callable:

    comfy_runner(workflow: dict, images: list[{"name", "image"}]) -> bytes (PNG)

The real runner (handler.py) goes through the official worker-comfyui
handler; tests pass a fake that returns a synthetic generation.
"""

from __future__ import annotations

import base64
import re
import secrets
import time
from typing import Callable

import numpy as np

from . import imaging
from .imaging import InputError
from .workflow import MODEL_FILES, WorkflowParams, build_workflow

ComfyRunner = Callable[[dict, list], bytes]

VERSION = "1.1.0"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _safe_id(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(job_id))[:64] or "job"


def resolve_alpha_mode(mode: str, has_alpha_image: bool) -> str:
    """'auto' -> 'provided' with a silhouette, else 'key_add'."""
    if mode == "auto":
        return "provided" if has_alpha_image else "key_add"
    if mode in ("provided", "provided_glow") and not has_alpha_image:
        raise InputError(f"alpha_mode '{mode}' needs alpha_image")
    return mode


def run_job(job_id: str, inp: dict, comfy_runner: ComfyRunner) -> dict:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # ---- decode inputs ----------------------------------------------------
    src = imaging.load_source_png(imaging.fetch_bytes(inp["image"], "image"))
    size = src.size
    mask = imaging.load_mask(imaging.fetch_bytes(inp["mask"], "mask"), size)
    if not (mask > 0).any():
        raise InputError("mask: no pixels selected (mask is empty)")

    guide = None
    if inp.get("guide_image"):
        guide = imaging.load_guide(imaging.fetch_bytes(inp["guide_image"], "guide_image"), size)

    provided_alpha = None
    if inp.get("alpha_image"):
        provided_alpha = imaging.load_alpha_image(imaging.fetch_bytes(inp["alpha_image"], "alpha_image"), size)
    alpha_mode = resolve_alpha_mode(inp["alpha_mode"], provided_alpha is not None)

    matte = imaging.parse_hex_color(inp["matte_color"])
    seed = inp["seed"] if inp["seed"] >= 0 else secrets.randbelow(2**53)
    timings["decode_s"] = time.perf_counter() - t0

    # ---- plan the crop ----------------------------------------------------
    model_region = imaging.grow_mask(mask, inp["mask_grow_px"])
    plan = imaging.plan_crop(
        model_region,
        pad_px=inp["context_padding_px"],
        pad_ratio=inp["context_padding_ratio"],
        work_megapixels=inp["work_megapixels"],
        max_upscale=inp["max_upscale"],
    )
    ys, xs = plan.slice()
    work = (plan.work_w, plan.work_h)

    src_work = imaging.resize_rgb(imaging.flatten_on_matte(src.rgba[ys, xs], matte), work)
    mask_work = imaging.resize_mask_binary(model_region[ys, xs], work)
    guide_work = imaging.resize_rgb(guide[ys, xs], work) if guide is not None else None

    sid = _safe_id(job_id)
    names = {
        "source": f"sprite_inpaint_{sid}_source.png",
        "mask": f"sprite_inpaint_{sid}_mask.png",
        "guide": f"sprite_inpaint_{sid}_guide.png",
    }
    uploads = [
        {"name": names["source"], "image": _b64(imaging.png_bytes(src_work, "RGB"))},
        {"name": names["mask"], "image": _b64(imaging.png_bytes(np.stack([mask_work] * 3, -1), "RGB"))},
    ]
    if guide_work is not None:
        uploads.append({"name": names["guide"], "image": _b64(imaging.png_bytes(guide_work, "RGB"))})

    workflow = build_workflow(WorkflowParams(
        source_name=names["source"],
        mask_name=names["mask"],
        guide_name=names["guide"] if guide_work is not None else None,
        guide_preprocess=inp["guide_preprocess"],
        prompt=inp["prompt"],
        negative_prompt=inp["negative_prompt"],
        seed=seed,
        steps=inp["steps"],
        cfg=inp["cfg"],
        sampler_name=inp["sampler_name"],
        scheduler=inp["scheduler"],
        denoise=inp["denoise"],
        shift=inp["shift"],
        control_strength=inp["control_strength"],
        canny_low=inp["canny_low"],
        canny_high=inp["canny_high"],
        filename_prefix=f"sprite_inpaint/{sid}",
        diffusion_precision=inp["diffusion_precision"],
    ))
    timings["prepare_s"] = time.perf_counter() - t0 - timings["decode_s"]

    # ---- generate ---------------------------------------------------------
    t1 = time.perf_counter()
    raw_png = comfy_runner(workflow, uploads)
    timings["comfy_s"] = time.perf_counter() - t1

    # ---- composite --------------------------------------------------------
    t2 = time.perf_counter()
    gen_work = imaging.decode_generated(raw_png)
    gen_crop = imaging.resize_rgb(gen_work, (plan.w, plan.h))
    gen_rgb, gen_a = imaging.recover_rgba(
        gen_crop, src.rgba[ys, xs], matte,
        mode=alpha_mode, knee=inp["alpha_key_knee"],
        provided_alpha=provided_alpha[ys, xs] if provided_alpha is not None else None,
    )
    weight = imaging.composite_weight(mask[ys, xs], inp["feather_px"])
    result = imaging.composite(src.rgba, plan, gen_rgb, gen_a, weight)

    meta = {
        "worker": f"refactor-sprite-inpaint {VERSION}",
        "prompt": inp["prompt"],
        "seed": seed,
        "steps": inp["steps"],
        "cfg": inp["cfg"],
        "denoise": inp["denoise"],
        "control_strength": inp["control_strength"],
        "guide": guide is not None,
        "alpha_mode": alpha_mode,
        "diffusion_precision": inp["diffusion_precision"],
        "crop_box": plan.box,
        "models": MODEL_FILES,
    }
    text = imaging.metadata_text(meta)
    if inp["return_format"] == "patch":
        out_png = imaging.png_bytes(result.rgba[ys, xs], "RGBA", icc_profile=src.icc_profile, text=text)
    else:
        out_png = imaging.png_bytes(result.rgba, "RGBA", icc_profile=src.icc_profile, text=text)
    timings["composite_s"] = time.perf_counter() - t2
    timings["total_s"] = time.perf_counter() - t0

    out = {
        "image": _b64(out_png),
        "format": inp["return_format"],
        "width": size[0],
        "height": size[1],
        "crop_box": plan.box,
        "work_size": [plan.work_w, plan.work_h],
        "seed": seed,
        "alpha_mode": alpha_mode,
        "diffusion_precision": inp["diffusion_precision"],
        "masked_pixels": result.masked_pixels,
        "unmasked_pixels_changed": result.unmasked_changed,
        "models": MODEL_FILES,
        "worker_version": VERSION,
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    if inp["return_debug"]:
        out["debug"] = {
            "model_source": uploads[0]["image"],
            "model_mask": uploads[1]["image"],
            "model_guide": uploads[2]["image"] if guide_work is not None else None,
            "raw_generation": _b64(raw_png),
        }
    return out
