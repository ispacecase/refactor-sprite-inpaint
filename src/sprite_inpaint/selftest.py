"""CPU-only self-test run during `docker build` (and by the unit tests).

Checks that the workflow template references exactly the models baked into
the image, that the pruned graphs are consistent, and that compositing keeps
unmasked RGBA bytes identical (including partial-alpha glow pixels).
Exit code 0 = OK.
"""

from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np
from PIL import Image

from . import pipeline, workflow
from .schema import FIELDS


def synthetic_sprite(w: int = 320, h: int = 256) -> np.ndarray:
    """A transparent canvas with an opaque magenta 'limb' and a soft cyan glow."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgba = np.zeros((h, w, 4), np.float32)
    # soft glow (partial alpha everywhere it is visible)
    d = np.hypot(xx - w * 0.5, yy - h * 0.5)
    glow = np.clip(1.0 - d / (min(w, h) * 0.45), 0, 1) ** 2
    rgba[..., 1] = 1.0
    rgba[..., 2] = 1.0
    rgba[..., 3] = glow * 0.6
    # opaque limb
    limb = (np.abs(yy - h * 0.5) < h * 0.08) & (xx > w * 0.2) & (xx < w * 0.8)
    rgba[limb] = [1.0, 0.1, 0.8, 1.0]
    out = np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8)
    # hidden colour under fully transparent pixels must survive too
    out[(out[..., 3] == 0)] = [17, 34, 51, 0]
    return out


def default_input(**overrides) -> dict:
    inp = {f.name: f.default for f in FIELDS if not f.required}
    inp.update(overrides)
    return inp


def png_b64(arr: np.ndarray, mode: str) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def fake_runner(color=(255, 140, 0)):
    """Stands in for ComfyUI: returns a solid generation at the work size."""
    seen = {}

    def run(wf: dict, images: list) -> bytes:
        seen["workflow"] = wf
        seen["images"] = images
        src = Image.open(io.BytesIO(base64.b64decode(images[0]["image"])))
        gen = Image.new("RGB", src.size, color)
        buf = io.BytesIO()
        gen.save(buf, "PNG")
        return buf.getvalue()

    run.seen = seen
    return run


def check_template() -> None:
    tpl = workflow.load_template()
    names = {
        tpl["1"]["inputs"]["unet_name"]: "diffusion_models",
        tpl["2"]["inputs"]["clip_name"]: "text_encoders",
        tpl["3"]["inputs"]["vae_name"]: "vae",
        tpl["4"]["inputs"]["name"]: "model_patches",
    }
    for fname, folder in names.items():
        assert workflow.MODEL_FILES[folder] == fname, (folder, fname)
    models_dir = os.environ.get("SELFTEST_MODELS_DIR")
    if models_dir:  # inside the image: the files must really be there
        for folder, fname in workflow.MODEL_FILES.items():
            path = os.path.join(models_dir, folder, fname)
            assert os.path.getsize(path) > 0, path


def check_composite() -> None:
    sprite = synthetic_sprite()
    h, w = sprite.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[100:160, 60:140] = 255
    run = fake_runner()
    inp = default_input(image=png_b64(sprite, "RGBA"), mask=png_b64(mask, "L"),
                        prompt="repair the arm", seed=7)
    out = pipeline.run_job("selftest", inp, run)
    assert out["unmasked_pixels_changed"] == 0
    res = np.array(Image.open(io.BytesIO(base64.b64decode(out["image"]))).convert("RGBA"))
    keep = mask == 0
    assert np.array_equal(res[keep], sprite[keep]), "unmasked pixels differ"
    assert not np.array_equal(res[~keep], sprite[~keep]), "masked area was not changed"
    wf = run.seen["workflow"]
    assert "12" not in wf and "image" not in wf["20"]["inputs"]


def main() -> int:
    check_template()
    check_composite()
    print("sprite_inpaint selftest: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
