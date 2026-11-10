"""Pixel maths for sprite repair: decoding, masks, crop planning, alpha and
exact compositing.

Nothing here imports ComfyUI, torch or RunPod, so it runs (and is tested) on a
plain CPU. All arrays are HxWxC numpy arrays; uint8 at the edges, float32 in
between.

The core guarantee lives in ``composite``: every pixel whose mask value is 0
is copied byte-for-byte from the original RGBA, and the function verifies it.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import re
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image, PngImagePlugin
from scipy import ndimage

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DATA_URI = re.compile(r"^data:[^,]*?;base64,", re.IGNORECASE)
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 8192 * 8192


class InputError(ValueError):
    """Bad job input; reported to the caller as-is."""


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def fetch_bytes(value: str, field: str) -> bytes:
    """Decode a base64 string / data URI, or download an https:// URL."""
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field}: empty")
    v = value.strip()
    if v.lower().startswith("https://"):
        try:
            req = urllib.request.Request(v, headers={"User-Agent": "refactor-sprite-inpaint/1"})
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (https only)
                data = resp.read(MAX_DOWNLOAD_BYTES + 1)
        except Exception as exc:  # pragma: no cover - network
            raise InputError(f"{field}: download failed ({exc})") from exc
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise InputError(f"{field}: download larger than {MAX_DOWNLOAD_BYTES} bytes")
        return data
    if v.lower().startswith("http://"):
        raise InputError(f"{field}: only https:// URLs are accepted")
    s = "".join(_DATA_URI.sub("", v).split())
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputError(f"{field}: not valid base64 ({exc})") from exc


def _open(data: bytes, field: str) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > MAX_PIXELS:
            raise InputError(f"{field}: {im.width}x{im.height} is larger than the {MAX_PIXELS} pixel limit")
        im.load()
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"{field}: not a readable image ({exc})") from exc
    return im


def _to_rgba(im: Image.Image) -> Image.Image:
    # Pillow applies tRNS transparency (P, L and RGB modes) when converting.
    return im if im.mode == "RGBA" else im.convert("RGBA")


@dataclass
class SourceImage:
    rgba: np.ndarray  # HxWx4 uint8
    icc_profile: bytes | None
    had_alpha: bool

    @property
    def size(self) -> tuple[int, int]:
        return self.rgba.shape[1], self.rgba.shape[0]


def load_source_png(data: bytes) -> SourceImage:
    """The sprite. Must be an 8-bit PNG so unmasked bytes round-trip exactly."""
    if not data.startswith(PNG_SIGNATURE):
        raise InputError("image: must be a PNG")
    if len(data) > 25 and data[12:16] == b"IHDR" and data[24] == 16:
        raise InputError("image: 16-bit PNGs are not supported (pixels would not round-trip exactly); export 8-bit")
    im = _open(data, "image")
    had_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        raise InputError(f"image: unsupported PNG mode {im.mode}; export 8-bit RGBA")
    rgba = np.array(_to_rgba(im), dtype=np.uint8)
    return SourceImage(rgba=rgba, icc_profile=im.info.get("icc_profile"), had_alpha=had_alpha)


def load_mask(data: bytes, size: tuple[int, int], field: str = "mask") -> np.ndarray:
    """Float32 HxW in [0, 1]: max(R, G, B) x alpha."""
    im = _open(data, field)
    if im.size != size:
        raise InputError(f"{field}: size {im.size[0]}x{im.size[1]} does not match image {size[0]}x{size[1]}")
    rgba = np.asarray(_to_rgba(im), dtype=np.float32) / 255.0
    return rgba[..., :3].max(axis=-1) * rgba[..., 3]


def load_alpha_image(data: bytes, size: tuple[int, int]) -> np.ndarray:
    """Target silhouette: alpha channel if present, otherwise brightness."""
    im = _open(data, "alpha_image")
    if im.size != size:
        raise InputError(f"alpha_image: size {im.size[0]}x{im.size[1]} does not match image {size[0]}x{size[1]}")
    has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    rgba = np.asarray(_to_rgba(im), dtype=np.float32) / 255.0
    if has_alpha:
        return rgba[..., 3]
    return rgba[..., :3].max(axis=-1)


def load_guide(data: bytes, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 at the sprite's size. Same aspect (within 1%) is resized."""
    im = _open(data, "guide_image")
    if im.size != size:
        ga, ia = im.size[0] / im.size[1], size[0] / size[1]
        if abs(ga - ia) / ia > 0.01:
            raise InputError(
                f"guide_image: {im.size[0]}x{im.size[1]} has a different aspect ratio from image {size[0]}x{size[1]}")
        im = im.convert("RGB").resize(size, Image.LANCZOS)
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        # Control maps are drawn on black; transparent areas become black.
        rgba = np.asarray(_to_rgba(im), dtype=np.float32)
        rgb = rgba[..., :3] * (rgba[..., 3:4] / 255.0)
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return np.array(im.convert("RGB"), dtype=np.uint8)


def parse_hex_color(value: str) -> np.ndarray:
    v = value.lstrip("#")
    return np.array([int(v[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Mask geometry
# ---------------------------------------------------------------------------

def grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    """Boolean mask dilated by a Euclidean radius (for the model only)."""
    inside = mask > 0
    if grow_px <= 0 or not inside.any():
        return inside
    dist_out = ndimage.distance_transform_edt(~inside)
    return dist_out <= grow_px


def composite_weight(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Blend weight that is exactly 0 wherever ``mask`` is 0.

    The feather ramps *inward* from the mask edge, so softening never touches
    pixels outside the mask. Image borders are not treated as edges.
    """
    inside = mask > 0
    if feather_px <= 0:
        return np.where(inside, mask, 0.0).astype(np.float32)
    # Pad with "inside" so the image border does not count as an edge.
    padded = np.pad(inside, 1, mode="constant", constant_values=True)
    dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    ramp = np.clip(dist / float(feather_px), 0.0, 1.0)
    return np.where(inside, mask * ramp, 0.0).astype(np.float32)


@dataclass
class CropPlan:
    x0: int
    y0: int
    x1: int
    y1: int
    work_w: int
    work_h: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def box(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    def slice(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


def plan_crop(region: np.ndarray, *, pad_px: int, pad_ratio: float, work_megapixels: float,
              max_upscale: float, multiple: int = 16) -> CropPlan:
    """Crop around ``region`` (bool HxW) with context, and pick a model size.

    The model size keeps the crop's aspect, targets ``work_megapixels`` and is
    rounded to ``multiple`` (Z-Image needs multiples of 16).
    """
    if not region.any():
        raise InputError("mask: no pixels selected (mask is empty)")
    h, w = region.shape
    ys = np.flatnonzero(region.any(axis=1))
    xs = np.flatnonzero(region.any(axis=0))
    bx0, bx1, by0, by1 = int(xs[0]), int(xs[-1]) + 1, int(ys[0]), int(ys[-1]) + 1
    pad = max(int(pad_px), int(round(pad_ratio * max(bx1 - bx0, by1 - by0))))
    x0, y0 = max(0, bx0 - pad), max(0, by0 - pad)
    x1, y1 = min(w, bx1 + pad), min(h, by1 + pad)
    cw, ch = x1 - x0, y1 - y0
    scale = math.sqrt(work_megapixels * 1_000_000 / (cw * ch))
    scale = min(scale, max_upscale)
    work_w = max(multiple, int(round(cw * scale / multiple)) * multiple)
    work_h = max(multiple, int(round(ch * scale / multiple)) * multiple)
    return CropPlan(x0, y0, x1, y1, work_w, work_h)


# ---------------------------------------------------------------------------
# Model-space images
# ---------------------------------------------------------------------------

def flatten_on_matte(rgba: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """uint8 RGBA -> uint8 RGB composited over a solid matte colour."""
    f = rgba.astype(np.float32) / 255.0
    a = f[..., 3:4]
    rgb = f[..., :3] * a + matte * (1.0 - a)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb
    return np.array(Image.fromarray(rgb, "RGB").resize(size, Image.LANCZOS), dtype=np.uint8)


def resize_mask_binary(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bool mask -> uint8 0/255 at ``size``; any coverage counts (conservative)."""
    im = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    arr = np.asarray(im)
    return np.where(arr > 2, 255, 0).astype(np.uint8)


def png_bytes(arr: np.ndarray, mode: str, *, icc_profile: bytes | None = None,
              text: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    kwargs = {"format": "PNG", "compress_level": 6}
    if icc_profile:
        kwargs["icc_profile"] = icc_profile
    if text:
        info = PngImagePlugin.PngInfo()
        for k, v in text.items():
            info.add_itxt(k, v)
        kwargs["pnginfo"] = info
    Image.fromarray(arr, mode).save(buf, **kwargs)
    return buf.getvalue()


def decode_generated(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data))
    im.load()
    return np.array(im.convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Alpha recovery and compositing
# ---------------------------------------------------------------------------

def recover_rgba(gen_rgb: np.ndarray, orig_rgba: np.ndarray, matte: np.ndarray, *, mode: str,
                 knee: float, provided_alpha: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Turn a generation made on ``matte`` back into straight colour + alpha.

    Returns (rgb float32 HxWx3 in 0..1, alpha float32 HxW in 0..1).
    The keyed alpha treats the matte as "fully transparent" and anything at
    least ``knee`` away from it (max channel distance) as opaque, which keeps
    partial-alpha neon glows instead of thresholding them away.
    """
    g = gen_rgb.astype(np.float32) / 255.0
    orig_a = orig_rgba[..., 3].astype(np.float32) / 255.0
    keyed = np.clip(np.abs(g - matte).max(axis=-1) / knee, 0.0, 1.0)
    if mode == "preserve":
        a = orig_a
    elif mode == "key":
        a = keyed
    elif mode == "key_add":
        a = np.maximum(orig_a, keyed)
    elif mode in ("provided", "provided_glow"):
        if provided_alpha is None:
            raise InputError(f"alpha_mode '{mode}' needs alpha_image")
        a = provided_alpha if mode == "provided" else np.maximum(provided_alpha, keyed)
    else:  # pragma: no cover - schema rejects this
        raise InputError(f"unknown alpha_mode {mode}")
    a = a.astype(np.float32)
    # Un-matte: C = a*F + (1-a)*M  =>  F = M + (C - M) / a
    safe = np.maximum(a, 1.0 / 255.0)[..., None]
    rgb = np.clip(matte + (g - matte) / safe, 0.0, 1.0)
    rgb = np.where(a[..., None] > 0, rgb, 0.0).astype(np.float32)
    return rgb, a


@dataclass
class CompositeResult:
    rgba: np.ndarray  # HxWx4 uint8, full image
    unmasked_changed: int
    masked_pixels: int


def composite(orig: np.ndarray, plan: CropPlan, gen_rgb: np.ndarray, gen_a: np.ndarray,
              weight: np.ndarray) -> CompositeResult:
    """Blend the generated crop into ``orig`` in premultiplied space.

    ``weight`` is the crop-sized composite weight. Where it is 0 the original
    bytes are copied verbatim; this is verified before returning.
    """
    out = orig.copy()
    ys, xs = plan.slice()
    o = orig[ys, xs].astype(np.float32) / 255.0
    w = weight[..., None]
    oa = o[..., 3:4]
    ga = gen_a[..., None]
    premul = w * (gen_rgb * ga) + (1.0 - w) * (o[..., :3] * oa)
    alpha = w * ga + (1.0 - w) * oa
    rgb = np.where(alpha > 0, premul / np.maximum(alpha, 1e-8), 0.0)
    blended = np.concatenate([rgb, alpha], axis=-1)
    blended = np.clip(np.rint(blended * 255.0), 0, 255).astype(np.uint8)
    keep = weight <= 0
    blended[keep] = orig[ys, xs][keep]
    out[ys, xs] = blended

    full_weight = np.zeros(orig.shape[:2], dtype=bool)
    full_weight[ys, xs] = weight > 0
    changed = int(np.any(out[~full_weight] != orig[~full_weight], axis=-1).sum())
    if changed:
        raise RuntimeError(f"internal error: {changed} unmasked pixels changed during compositing")
    return CompositeResult(rgba=out, unmasked_changed=changed, masked_pixels=int(full_weight.sum()))


def metadata_text(params: dict) -> dict[str, str]:
    return {"refactor-sprite-inpaint": json.dumps(params, sort_keys=True)}
