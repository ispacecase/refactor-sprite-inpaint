"""handler.py against a stub of the official worker-comfyui handler."""

import base64
import importlib
import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

from sprite_inpaint.selftest import png_b64, synthetic_sprite


class StubUpstream(types.ModuleType):
    def __init__(self):
        super().__init__("comfy_worker_handler")
        self.calls = []
        self.response = None

    def handler(self, job):
        self.calls.append(job)
        if self.response is not None:
            return self.response
        src = Image.open(io.BytesIO(base64.b64decode(job["input"]["images"][0]["image"])))
        buf = io.BytesIO()
        Image.new("RGB", src.size, (0, 255, 200)).save(buf, "PNG")
        return {"images": [{"filename": "x_00001_.png", "type": "base64",
                            "data": base64.b64encode(buf.getvalue()).decode()}]}


@pytest.fixture
def env(monkeypatch, tmp_path):
    stub = StubUpstream()
    monkeypatch.setitem(sys.modules, "comfy_worker_handler", stub)
    monkeypatch.setenv("COMFY_DIR", str(tmp_path))
    monkeypatch.delenv("BUCKET_ENDPOINT_URL", raising=False)
    (tmp_path / "input").mkdir()
    (tmp_path / "output" / "sprite_inpaint").mkdir(parents=True)
    handler = importlib.reload(importlib.import_module("handler"))
    return handler, stub, tmp_path


def job(**over):
    sprite = synthetic_sprite(160, 128)
    mask = np.zeros(sprite.shape[:2], np.uint8)
    mask[40:90, 30:80] = 255
    inp = {"image": png_b64(sprite, "RGBA"), "mask": png_b64(mask, "L"), "prompt": "repair forearm"}
    inp.update(over)
    return {"id": "job-1", "input": inp}, sprite, mask


def test_success_and_exactness(env):
    handler, stub, tmp = env
    (tmp / "input" / "sprite_inpaint_job-1_source.png").write_bytes(b"x")
    (tmp / "output" / "sprite_inpaint" / "x_00001_.png").write_bytes(b"x")
    j, sprite, mask = job(seed=5)
    out = handler.handler(j)
    assert "error" not in out, out
    assert out["seed"] == 5 and out["unmasked_pixels_changed"] == 0
    res = np.array(Image.open(io.BytesIO(base64.b64decode(out["image"]))))
    assert res.shape == sprite.shape
    assert np.array_equal(res[mask == 0], sprite[mask == 0])
    sent = stub.calls[0]["input"]
    assert {i["name"] for i in sent["images"]} == {"sprite_inpaint_job-1_source.png", "sprite_inpaint_job-1_mask.png"}
    assert sent["workflow"]["50"]["inputs"]["seed"] == 5
    # uploads and outputs cleaned up
    assert not list((tmp / "input").iterdir())
    assert not list((tmp / "output" / "sprite_inpaint").iterdir())


def test_guide_and_patch_and_debug(env):
    handler, stub, _ = env
    j, sprite, _ = job()
    guide = np.zeros((128, 160, 3), np.uint8)
    guide[60, 20:140] = 255
    j["input"].update(guide_image=png_b64(guide, "RGB"), return_format="patch", return_debug=True)
    out = handler.handler(j)
    assert "error" not in out, out
    x0, y0, x1, y1 = out["crop_box"]
    patch = Image.open(io.BytesIO(base64.b64decode(out["image"])))
    assert patch.size == (x1 - x0, y1 - y0)
    assert out["debug"]["model_guide"] and out["debug"]["raw_generation"]
    wf = stub.calls[0]["input"]["workflow"]
    assert wf["20"]["inputs"]["image"] == ["12", 0]
    assert 0 <= out["seed"] < 2**53


def test_invalid_input_is_reported(env):
    handler, stub, _ = env
    out = handler.handler({"id": "j", "input": {"image": "x", "prompt": "p", "steps": 500}})
    assert out["error"] == "Invalid input"
    assert any("mask" in d for d in out["details"])
    assert any("steps" in d and "range" in d for d in out["details"])
    assert not stub.calls


def test_bad_image_and_empty_mask(env):
    handler, _, _ = env
    j, _, _ = job()
    j["input"]["image"] = base64.b64encode(b"nope").decode()
    assert "must be a PNG" in handler.handler(j)["error"]
    j, sprite, _ = job()
    j["input"]["mask"] = png_b64(np.zeros(sprite.shape[:2], np.uint8), "L")
    assert "empty" in handler.handler(j)["error"]


def test_provided_alpha_requires_image(env):
    handler, _, _ = env
    j, _, _ = job(alpha_mode="provided")
    assert "alpha_image" in handler.handler(j)["error"]


def test_upstream_error_passthrough(env):
    handler, stub, _ = env
    stub.response = {"error": "Workflow validation failed", "details": ["Value not in list"]}
    j, _, _ = job()
    out = handler.handler(j)
    assert out["error"].startswith("ComfyUI failed") and out["details"] == ["Value not in list"]


def test_s3_env_is_refused(env, monkeypatch):
    handler, stub, _ = env
    monkeypatch.setenv("BUCKET_ENDPOINT_URL", "https://bucket.example")
    j, _, _ = job()
    assert "BUCKET_ENDPOINT_URL" in handler.handler(j)["error"]
    assert not stub.calls


def test_auto_uses_provided_silhouette(env):
    """With alpha_image and no alpha_mode, repainted alpha is the silhouette."""
    handler, _, _ = env
    j, sprite, mask = job()
    sil = np.zeros(sprite.shape[:2] + (4,), np.uint8)
    sil[50:70, 40:60] = [255, 255, 255, 255]       # opaque square inside the mask
    j["input"]["alpha_image"] = png_b64(sil, "RGBA")
    j["input"]["feather_px"] = 0
    out = handler.handler(j)
    assert "error" not in out, out
    assert out["alpha_mode"] == "provided"
    res = np.array(Image.open(io.BytesIO(base64.b64decode(out["image"]))))
    m = mask > 0
    assert np.array_equal(res[..., 3][m], sil[..., 3][m])
    assert np.array_equal(res[~m], sprite[~m])


def test_auto_without_silhouette_is_key_add(env):
    handler, _, _ = env
    j, _, _ = job()
    assert handler.handler(j)["alpha_mode"] == "key_add"


def test_fp8_precision_sets_weight_dtype(env):
    handler, stub, _ = env
    j, _, _ = job(diffusion_precision="fp8")
    out = handler.handler(j)
    assert out["diffusion_precision"] == "fp8"
    assert stub.calls[0]["input"]["workflow"]["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"
    j, _, _ = job()
    handler.handler(j)
    assert stub.calls[1]["input"]["workflow"]["1"]["inputs"]["weight_dtype"] == "default"
