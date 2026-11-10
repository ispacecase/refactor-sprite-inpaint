"""Job input schema for the refactor-sprite-inpaint endpoint.

This module is the single source of truth. It produces:

* ``RUNPOD_SCHEMA`` - the dict format understood by RunPod's
  ``runpod.serverless.utils.rp_validator.validate`` (type / required /
  default / constraints).
* ``json_schema()`` - an equivalent JSON Schema (draft 2020-12) that is
  written to ``schemas/input.schema.json`` for clients and tooling.
  ``tests/test_schema.py`` fails if the committed file drifts from this.

Inputs are deliberately flat (no nested objects) because RunPod's validator
only checks top-level keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MAX_SEED = 2**53 - 1  # JSON-safe in every client language

SAMPLERS = (
    "res_multistep",
    "euler",
    "euler_ancestral",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_sde",
    "uni_pc",
    "ddim",
    "lcm",
)
SCHEDULERS = (
    "simple",
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "beta",
    "linear_quadratic",
    "ddim_uniform",
)
ALPHA_MODES = ("auto", "provided", "key_add", "preserve", "key", "provided_glow")
PRECISIONS = ("bf16", "fp8")
GUIDE_PREPROCESS = ("none", "canny")
RETURN_FORMATS = ("full", "patch")

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class Field:
    name: str
    type: type
    description: str
    required: bool = False
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[tuple] = None
    pattern: Optional[str] = None
    max_length: Optional[int] = None
    nullable: bool = False
    extra_check: Optional[Callable[[Any], bool]] = field(default=None, compare=False)

    def check(self, value: Any) -> bool:
        if value is None:
            return self.nullable or not self.required
        if self.type in (int, float) and isinstance(value, bool):
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        if self.choices is not None and value not in self.choices:
            return False
        if self.pattern is not None and not re.fullmatch(self.pattern, value):
            return False
        if self.max_length is not None and len(value) > self.max_length:
            return False
        if self.extra_check is not None and not self.extra_check(value):
            return False
        return True

    def hint(self) -> str:
        parts = [self.type.__name__]
        if self.choices:
            parts.append("one of " + ", ".join(map(str, self.choices)))
        if self.minimum is not None or self.maximum is not None:
            parts.append(f"range [{self.minimum}, {self.maximum}]")
        if self.pattern:
            parts.append(f"pattern {self.pattern}")
        if self.max_length:
            parts.append(f"max length {self.max_length}")
        return "; ".join(parts)


IMAGE_DESC = (
    "Base64 PNG (raw or data URI) or an https:// URL the worker downloads."
)

FIELDS: tuple[Field, ...] = (
    # ---- images -----------------------------------------------------------
    Field("image", str, required=True, description=(
        "The approved sprite. 8-bit PNG, RGBA or RGB. " + IMAGE_DESC)),
    Field("mask", str, required=True, description=(
        "Edit mask, same size as image. Non-black, non-transparent pixels are "
        "repainted; brightness sets strength (max(R,G,B) x alpha). Pixels with "
        "mask value 0 are returned byte-for-byte unchanged. " + IMAGE_DESC)),
    Field("guide_image", str, nullable=True, description=(
        "Optional pose/geometry guide, same size (or same aspect) as image: an "
        "OpenPose-style skeleton, scribble, line art, MLSD, depth or gray map "
        "drawn in the image's coordinates. " + IMAGE_DESC)),
    Field("alpha_image", str, nullable=True, description=(
        "Optional target silhouette for alpha_mode 'provided' / 'provided_glow'. "
        "Uses its alpha channel if it has one, otherwise its brightness. " + IMAGE_DESC)),
    # ---- prompt -----------------------------------------------------------
    Field("prompt", str, required=True, max_length=4000,
          extra_check=lambda v: bool(v.strip()),
          description="Correction prompt describing what the masked area should become."),
    Field("negative_prompt", str, default="", max_length=4000,
          description="Only used when cfg > 1 (Z-Image-Turbo ignores negatives at cfg 1)."),
    # ---- guide ------------------------------------------------------------
    Field("guide_preprocess", str, default="none", choices=GUIDE_PREPROCESS,
          description="'none' = guide is already a control map; 'canny' = run ComfyUI's Canny on it first."),
    Field("control_strength", float, default=0.9, minimum=0.0, maximum=2.0,
          description="Fun ControlNet strength (model card: 0.65-1.0; higher helps inpaint continuity)."),
    Field("canny_low", float, default=0.1, minimum=0.01, maximum=0.99,
          description="Canny low threshold (guide_preprocess 'canny' only)."),
    Field("canny_high", float, default=0.32, minimum=0.01, maximum=0.99,
          description="Canny high threshold (guide_preprocess 'canny' only)."),
    # ---- sampler ----------------------------------------------------------
    Field("seed", int, default=-1, minimum=-1, maximum=MAX_SEED,
          description="-1 picks a random seed; the seed used is returned."),
    Field("steps", int, default=8, minimum=1, maximum=50,
          description="Sampling steps (the 8-step distilled ControlNet is tuned for 8)."),
    Field("cfg", float, default=1.0, minimum=1.0, maximum=10.0,
          description="Classifier-free guidance. Z-Image-Turbo is distilled for 1.0."),
    Field("sampler_name", str, default="res_multistep", choices=SAMPLERS,
          description="ComfyUI sampler."),
    Field("scheduler", str, default="simple", choices=SCHEDULERS,
          description="ComfyUI scheduler."),
    Field("denoise", float, default=1.0, minimum=0.05, maximum=1.0,
          description="1.0 regenerates the masked area; lower keeps more of the original there."),
    Field("shift", float, default=3.0, minimum=0.0, maximum=20.0,
          description="ModelSamplingAuraFlow shift (Z-Image default 3)."),
    Field("diffusion_precision", str, default="bf16", choices=PRECISIONS,
          description=("Diffusion model weights in VRAM: bf16 (12.3 GB, best quality) or fp8 (about half; "
                       "use if a 24 GB GPU runs short or for faster cold starts).")),
    # ---- crop / resolution ------------------------------------------------
    Field("work_megapixels", float, default=1.0, minimum=0.25, maximum=2.0,
          description=("Area the cropped region is resized to for generation. Only the crop around the "
                       "mask is generated, never the whole sprite unless the mask spans it.")),
    Field("max_upscale", float, default=4.0, minimum=1.0, maximum=8.0,
          description="Cap on how much a small crop is enlarged for generation."),
    Field("context_padding_px", int, default=96, minimum=0, maximum=4096,
          description="Minimum context added around the mask bounding box, in source pixels."),
    Field("context_padding_ratio", float, default=0.35, minimum=0.0, maximum=4.0,
          description="Context added around the mask box as a fraction of its longer side (the larger of the two paddings wins)."),
    Field("mask_grow_px", int, default=6, minimum=0, maximum=256,
          description="Dilation of the mask the MODEL sees (source pixels). Does not widen what is composited."),
    Field("feather_px", int, default=3, minimum=0, maximum=128,
          description="Inward feather of the composite edge. Never extends outside the mask."),
    # ---- transparency -----------------------------------------------------
    Field("matte_color", str, default="#000000", pattern=_HEX_COLOR.pattern,
          description="Background the sprite is flattened onto for the model, and keyed out of the result."),
    Field("alpha_mode", str, default="auto", choices=ALPHA_MODES,
          description=(
              "Alpha of repainted pixels. auto (default) = 'provided' when alpha_image is given, "
              "otherwise 'key_add'. provided = alpha_image exactly (precise part repairs with a known "
              "silhouette); key_add = max(original alpha, keyed alpha) (extend limbs, keeps glows); "
              "preserve = original alpha; key = keyed alpha only (can erase); "
              "provided_glow = max(alpha_image, keyed alpha).")),
    Field("alpha_key_knee", float, default=0.2, minimum=0.01, maximum=1.0,
          description="Keyed alpha = clamp(max |pixel - matte| / knee). Lower = more opaque."),
    # ---- output -----------------------------------------------------------
    Field("return_format", str, default="full", choices=RETURN_FORMATS,
          description="'full' returns the whole PNG; 'patch' returns only the crop box (client pastes it back)."),
    Field("return_debug", bool, default=False,
          description="Also return the model-space source, mask, guide and raw generation."),
)

FIELDS_BY_NAME = {f.name: f for f in FIELDS}


def _runpod_rules(f: Field) -> dict:
    rules: dict[str, Any] = {"type": f.type, "required": f.required}
    if not f.required:
        rules["default"] = f.default
    rules["constraints"] = f.check
    return rules


RUNPOD_SCHEMA: dict[str, dict] = {f.name: _runpod_rules(f) for f in FIELDS}


def runpod_schema() -> dict[str, dict]:
    """A fresh copy (rp_validator mutates the schema it is given)."""
    return {f.name: _runpod_rules(f) for f in FIELDS}


def validate_input(raw: dict) -> tuple[dict | None, list[str]]:
    """RunPod's validator, then our constraints on every field.

    rp_validator skips the ``constraints`` check whenever a value has the same
    type as the field's default (it ``continue``s past it), so ranges and
    enums on optional fields would silently pass. We re-check them here.
    Returns (validated_input, []) or (None, errors).
    """
    from runpod.serverless.utils.rp_validator import validate

    if not isinstance(raw, dict):
        return None, ["input must be a JSON object"]
    res = validate(raw, runpod_schema())
    errors = list(res.get("errors", []))
    for f in FIELDS:  # check every supplied value, even if other fields failed
        if f.name not in raw:
            continue
        v = raw[f.name]
        if f.type is float and type(v) is int:
            v = float(v)
        if v is None or isinstance(v, f.type):  # wrong types are already reported
            if not f.check(v):
                errors.append(f"{f.name} does not meet the constraints.")
    if errors:
        return None, explain_errors(list(dict.fromkeys(errors)))
    return res["validated_input"], []


def explain_errors(errors: list[str]) -> list[str]:
    """Append the expected type/range to RunPod's terse constraint errors."""
    out = []
    for e in errors:
        name = e.split(" ", 1)[0]
        f = FIELDS_BY_NAME.get(name)
        if f is not None and "constraints" in e:
            e = f"{e} Expected {f.hint()}."
        out.append(e)
    return out


_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def json_schema() -> dict:
    props: dict[str, Any] = {}
    for f in FIELDS:
        p: dict[str, Any] = {"description": f.description}
        t = _JSON_TYPES[f.type]
        p["type"] = [t, "null"] if f.nullable else t
        if not f.required:
            p["default"] = f.default
        if f.minimum is not None:
            p["minimum"] = f.minimum
        if f.maximum is not None:
            p["maximum"] = f.maximum
        if f.choices is not None:
            p["enum"] = list(f.choices) + ([None] if f.nullable else [])
        if f.pattern is not None:
            p["pattern"] = f.pattern
        if f.max_length is not None:
            p["maxLength"] = f.max_length
        if f.name == "prompt":
            p["minLength"] = 1
        props[f.name] = p
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "refactor-sprite-inpaint job input",
        "description": (
            "The object sent as {\"input\": ...} to the RunPod endpoint. Generated from "
            "src/sprite_inpaint/schema.py - do not edit by hand."),
        "type": "object",
        "additionalProperties": False,
        "required": [f.name for f in FIELDS if f.required],
        "properties": props,
    }


if __name__ == "__main__":  # python -m sprite_inpaint.schema > schemas/input.schema.json
    import json
    import sys

    json.dump(json_schema(), sys.stdout, indent=2)
    sys.stdout.write("\n")
