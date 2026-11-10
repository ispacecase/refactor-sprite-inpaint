import base64
import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from sprite_inpaint import imaging
from sprite_inpaint.imaging import InputError
from sprite_inpaint.selftest import synthetic_sprite


def _png(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, "PNG")
    return buf.getvalue()


def test_fetch_bytes_accepts_raw_and_data_uri():
    raw = b"\x89PNGhello"
    b64 = base64.b64encode(raw).decode()
    assert imaging.fetch_bytes(b64, "f") == raw
    assert imaging.fetch_bytes("data:image/png;base64," + b64, "f") == raw
    assert imaging.fetch_bytes(b64[:10] + "\n" + b64[10:], "f") == raw
    with pytest.raises(InputError):
        imaging.fetch_bytes("http://example.com/a.png", "f")
    with pytest.raises(InputError):
        imaging.fetch_bytes("not base64!!", "f")


def test_source_must_be_png_and_8bit():
    with pytest.raises(InputError):
        imaging.load_source_png(b"GIF89a....")
    # hand-build a 1x1 16-bit RGBA PNG
    ihdr = struct.pack(">IIBBBBB", 1, 1, 16, 6, 0, 0, 0)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    png = imaging.PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00" * 8)) + chunk(b"IEND", b"")
    with pytest.raises(InputError, match="16-bit"):
        imaging.load_source_png(png)


def test_palette_png_with_transparency_round_trips():
    sprite = synthetic_sprite(64, 48)
    pal = Image.fromarray(sprite, "RGBA").quantize(colors=64)
    buf = io.BytesIO()
    pal.save(buf, "PNG")
    src = imaging.load_source_png(buf.getvalue())
    assert src.rgba.shape == (48, 64, 4)
    assert np.array_equal(src.rgba, np.array(pal.convert("RGBA")))


def test_mask_uses_max_channel_times_alpha():
    m = np.zeros((4, 4, 4), np.uint8)
    m[0, 0] = [255, 255, 255, 255]   # white opaque -> 1
    m[0, 1] = [255, 0, 0, 255]       # red stroke counts -> 1
    m[0, 2] = [255, 255, 255, 0]     # white but transparent -> 0
    m[0, 3] = [128, 128, 128, 255]   # grey -> ~0.5
    mask = imaging.load_mask(_png(m, "RGBA"), (4, 4))
    assert mask[0, 0] == 1 and mask[0, 1] == 1 and mask[0, 2] == 0
    assert abs(mask[0, 3] - 128 / 255) < 1e-6
    with pytest.raises(InputError, match="does not match"):
        imaging.load_mask(_png(m, "RGBA"), (5, 4))


def test_composite_weight_is_zero_outside_and_feathers_inward():
    mask = np.zeros((40, 40), np.float32)
    mask[10:30, 10:30] = 1.0
    w = imaging.composite_weight(mask, 4)
    assert np.all(w[mask == 0] == 0)
    assert w[20, 20] == 1.0
    assert 0 < w[10, 20] < 1  # edge pixel is softened
    # touching the image border: no feather against the border
    mask2 = np.zeros((40, 40), np.float32)
    mask2[0:10, 0:10] = 1.0
    w2 = imaging.composite_weight(mask2, 4)
    assert w2[0, 0] == 1.0


def test_thin_mask_still_gets_weight():
    mask = np.zeros((20, 20), np.float32)
    mask[10, 2:18] = 1.0  # 1px wide line
    w = imaging.composite_weight(mask, 8)
    assert (w > 0).sum() == 16


def test_grow_mask():
    mask = np.zeros((21, 21), np.float32)
    mask[10, 10] = 1
    g = imaging.grow_mask(mask, 3)
    assert g[10, 13] and not g[10, 14] and g[10, 10]


def test_plan_crop_bounds_and_multiples():
    region = np.zeros((1000, 800), bool)
    region[400:500, 100:200] = True
    p = imaging.plan_crop(region, pad_px=96, pad_ratio=0.35, work_megapixels=1.0, max_upscale=4.0)
    assert 0 <= p.x0 < 100 and p.x1 > 200 and 0 <= p.y0 < 400 and p.y1 > 500
    assert p.x1 <= 800 and p.y1 <= 1000
    assert p.work_w % 16 == 0 and p.work_h % 16 == 0
    assert abs(p.work_w * p.work_h - 1_000_000) / 1_000_000 < 0.1
    tiny = np.zeros((64, 64), bool)
    tiny[30:34, 30:34] = True
    p2 = imaging.plan_crop(tiny, pad_px=8, pad_ratio=0, work_megapixels=1.0, max_upscale=4.0)
    assert p2.work_w <= p2.w * 4 + 16
    with pytest.raises(InputError):
        imaging.plan_crop(np.zeros((5, 5), bool), pad_px=0, pad_ratio=0, work_megapixels=1, max_upscale=1)


@pytest.mark.parametrize("mode", ["key", "key_add", "preserve", "provided", "provided_glow"])
@pytest.mark.parametrize("matte_hex", ["#000000", "#00ff00"])
def test_recovered_rgba_reflattens_to_generation(mode, matte_hex):
    """Whatever alpha a mode picks, colour is solved so that putting the
    result back on the matte reproduces the generated pixels (when the
    alpha can explain them)."""
    rng = np.random.default_rng(1)
    matte = imaging.parse_hex_color(matte_hex)
    gen = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    orig = rng.integers(0, 256, (32, 32, 4), dtype=np.uint8)
    orig[..., 3] = 255
    provided = np.ones((32, 32), np.float32)
    rgb, a = imaging.recover_rgba(gen, orig, matte, mode=mode, knee=0.2, provided_alpha=provided)
    assert rgb.min() >= 0 and rgb.max() <= 1 and a.min() >= 0 and a.max() <= 1
    back = rgb * a[..., None] + matte * (1 - a[..., None])
    assert np.abs(back - gen / 255.0).max() < 1.5 / 255


def test_key_alpha_on_black():
    matte = np.zeros(3, np.float32)
    gen = np.array([[[0, 0, 0], [255, 0, 255], [20, 25, 10]]], np.uint8)  # bg, neon, faint glow
    orig = np.zeros((1, 3, 4), np.uint8)
    rgb, a = imaging.recover_rgba(gen, orig, matte, mode="key", knee=0.2, provided_alpha=None)
    assert a[0, 0] == 0 and a[0, 1] == 1
    assert 0 < a[0, 2] < 1                   # glow keeps partial alpha
    assert np.allclose(rgb[0, 2] * a[0, 2], gen[0, 2] / 255, atol=1e-6)


def test_provided_mode_requires_alpha_image():
    with pytest.raises(InputError):
        imaging.recover_rgba(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 4), np.uint8),
                             np.zeros(3, np.float32), mode="provided", knee=0.2, provided_alpha=None)


def test_composite_is_exact_outside_mask_with_soft_mask_and_glow():
    sprite = synthetic_sprite(200, 160)
    h, w = sprite.shape[:2]
    mask = np.zeros((h, w), np.float32)
    mask[50:110, 40:120] = 0.6           # soft mask value
    mask[70:90, 60:100] = 1.0
    region = imaging.grow_mask(mask, 6)
    plan = imaging.plan_crop(region, pad_px=20, pad_ratio=0.2, work_megapixels=0.3, max_upscale=4)
    ys, xs = plan.slice()
    gen = np.full((plan.h, plan.w, 3), (255, 120, 0), np.uint8)
    rgb, a = imaging.recover_rgba(gen, sprite[ys, xs], np.zeros(3, np.float32),
                                  mode="key_add", knee=0.2, provided_alpha=None)
    weight = imaging.composite_weight(mask[ys, xs], 3)
    res = imaging.composite(sprite, plan, rgb, a, weight)
    keep = mask == 0
    assert res.unmasked_changed == 0
    assert np.array_equal(res.rgba[keep], sprite[keep])
    # partial-alpha glow pixels and hidden colours survive exactly
    glow = keep & (sprite[..., 3] > 0) & (sprite[..., 3] < 255)
    assert glow.sum() > 1000
    assert np.array_equal(res.rgba[glow], sprite[glow])
    hidden = (sprite[..., 3] == 0) & keep
    assert np.array_equal(res.rgba[hidden], sprite[hidden])
    # fully-weighted core is the generation
    assert np.all(res.rgba[80, 80, :3] == [255, 120, 0])
    assert res.rgba[80, 80, 3] == 255


def test_png_bytes_round_trip_rgba_and_icc():
    sprite = synthetic_sprite(30, 20)
    data = imaging.png_bytes(sprite, "RGBA", icc_profile=b"fake-icc", text={"k": "v"})
    im = Image.open(io.BytesIO(data))
    assert np.array_equal(np.array(im), sprite)
    assert im.info.get("icc_profile") == b"fake-icc"
    assert im.text.get("k") == "v"
