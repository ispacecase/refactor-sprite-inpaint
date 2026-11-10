"""Build the ComfyUI API-format workflow for one job from the template.

The template (workflows/sprite_inpaint_api.json) contains every optional
node; this module fills in values and prunes what a job does not use, so the
graph sent to ComfyUI never references an image that was not uploaded.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from importlib import resources

TEMPLATE_NAME = "sprite_inpaint_api.json"

# Model files the template expects. The Dockerfile downloads exactly these.
MODEL_FILES = {
    "diffusion_models": "z_image_turbo_bf16.safetensors",
    "text_encoders": "qwen_3_4b.safetensors",
    "vae": "ae.safetensors",
    "model_patches": "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
}

N_UNET, N_CLIP, N_VAE, N_PATCH = "1", "2", "3", "4"
WEIGHT_DTYPES = {"bf16": "default", "fp8": "fp8_e4m3fn"}  # UNETLoader weight_dtype
N_SOURCE, N_MASK, N_GUIDE, N_CANNY = "10", "11", "12", "13"
N_CONTROL, N_SHIFT = "20", "21"
N_POS, N_NEG_ZERO, N_NEG_TEXT = "30", "31", "32"
N_SAMPLER, N_SAVE = "50", "70"


def load_template() -> dict:
    text = resources.files("sprite_inpaint").joinpath("workflows", TEMPLATE_NAME).read_text()
    return json.loads(text)


@dataclass
class WorkflowParams:
    source_name: str
    mask_name: str
    guide_name: str | None
    guide_preprocess: str
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    shift: float
    control_strength: float
    canny_low: float
    canny_high: float
    filename_prefix: str
    diffusion_precision: str = "bf16"


def build_workflow(p: WorkflowParams, template: dict | None = None) -> dict:
    wf = copy.deepcopy(template if template is not None else load_template())

    wf[N_UNET]["inputs"]["weight_dtype"] = WEIGHT_DTYPES[p.diffusion_precision]
    wf[N_SOURCE]["inputs"]["image"] = p.source_name
    wf[N_MASK]["inputs"]["image"] = p.mask_name

    control = wf[N_CONTROL]["inputs"]
    control["strength"] = float(p.control_strength)
    if p.guide_name is None:
        del wf[N_GUIDE], wf[N_CANNY]
        control.pop("image", None)  # inpaint-only: node feeds a neutral control latent
    else:
        wf[N_GUIDE]["inputs"]["image"] = p.guide_name
        if p.guide_preprocess == "canny":
            wf[N_CANNY]["inputs"]["low_threshold"] = float(p.canny_low)
            wf[N_CANNY]["inputs"]["high_threshold"] = float(p.canny_high)
            control["image"] = [N_CANNY, 0]
        else:
            del wf[N_CANNY]
            control["image"] = [N_GUIDE, 0]

    wf[N_SHIFT]["inputs"]["shift"] = float(p.shift)
    wf[N_POS]["inputs"]["text"] = p.prompt

    sampler = wf[N_SAMPLER]["inputs"]
    if p.cfg > 1.0:
        wf[N_NEG_TEXT]["inputs"]["text"] = p.negative_prompt
        sampler["negative"] = [N_NEG_TEXT, 0]
        del wf[N_NEG_ZERO]
    else:
        sampler["negative"] = [N_NEG_ZERO, 0]
        del wf[N_NEG_TEXT]

    sampler.update(
        seed=int(p.seed),
        steps=int(p.steps),
        cfg=float(p.cfg),
        sampler_name=p.sampler_name,
        scheduler=p.scheduler,
        denoise=float(p.denoise),
    )
    wf[N_SAVE]["inputs"]["filename_prefix"] = p.filename_prefix
    _check_links(wf)
    return wf


def _check_links(wf: dict) -> None:
    for node_id, node in wf.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in wf:
                    raise AssertionError(f"node {node_id}.{name} links to missing node {value[0]}")
