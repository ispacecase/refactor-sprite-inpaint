import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

import sprite_inpaint_client as c
from sprite_inpaint.selftest import default_input, fake_runner, png_b64, synthetic_sprite
from sprite_inpaint import pipeline


@pytest.fixture
def files(tmp_path):
    sprite = synthetic_sprite(120, 96)
    mask = np.zeros(sprite.shape[:2], np.uint8)
    mask[30:60, 20:70] = 255
    sp, mp = tmp_path / "s.png", tmp_path / "m.png"
    Image.fromarray(sprite, "RGBA").save(sp)
    Image.fromarray(mask, "L").save(mp)
    return sprite, mask, sp, mp, tmp_path


def test_build_input_validates(files):
    _, _, sp, mp, _ = files
    inp = c.build_input(str(sp), str(mp), "fix", seed=3, steps=8)
    assert base64.b64decode(inp["image"]) == sp.read_bytes()
    with pytest.raises(c.ClientError, match="unknown field"):
        c.build_input(str(sp), str(mp), "fix", wrong=1)
    with pytest.raises(c.ClientError, match="steps"):
        c.build_input(str(sp), str(mp), "fix", steps=0)
    with pytest.raises(c.ClientError, match="alpha_mode"):
        c.build_input(str(sp), str(mp), "fix", alpha_mode="nope")
    assert c.build_input("https://x/a.png", str(mp), "fix")["image"] == "https://x/a.png"


def test_build_request_cli_sends_nothing(files, monkeypatch):
    _, _, sp, mp, tmp = files
    monkeypatch.setattr(c.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    rc = c.main(["build-request", "--image", str(sp), "--mask", str(mp), "--prompt", "fix",
                 "--set", "denoise=0.8", "--seed", "9", "--out", str(tmp / "req.json")])
    assert rc == 0
    req = json.loads((tmp / "req.json").read_text())
    assert req["input"]["denoise"] == 0.8 and req["input"]["seed"] == 9


def test_patch_result_reassembles_to_full(files, tmp_path):
    sprite, mask, sp, mp, tmp = files
    base = default_input(image=png_b64(sprite, "RGBA"), mask=png_b64(mask, "L"), prompt="p", seed=1)
    full = pipeline.run_job("a", dict(base), fake_runner())
    patch = pipeline.run_job("a", dict(base, return_format="patch"), fake_runner())
    c.save_result(full, str(tmp / "full.png"), str(sp))
    c.save_result(patch, str(tmp / "patch.png"), str(sp))
    a = np.array(Image.open(tmp / "full.png").convert("RGBA"))
    b = np.array(Image.open(tmp / "patch.png").convert("RGBA"))
    assert np.array_equal(a, b)
    assert json.loads((tmp / "full.json").read_text())["seed"] == 1
    changed, masked = c.verify(str(sp), str(mp), str(tmp / "patch.png"))
    assert changed == 0 and masked == int((mask > 0).sum())


def test_verify_detects_changes(files):
    sprite, _, sp, mp, tmp = files
    bad = sprite.copy()
    bad[0, 0, 3] ^= 1
    Image.fromarray(bad, "RGBA").save(tmp / "bad.png")
    assert c.verify(str(sp), str(mp), str(tmp / "bad.png"))[0] == 1


def test_submit_size_limit_and_flow(monkeypatch):
    cl = c.SpriteInpaintClient("ep", "key")
    with pytest.raises(c.ClientError, match="over RunPod"):
        cl.submit({"image": "x" * (11 * 1024 * 1024)})
    calls = []
    statuses = iter([{"id": "j1", "status": "IN_QUEUE"},
                     {"id": "j1", "status": "COMPLETED", "output": {"image": "", "seed": 1}}])

    def fake_call(method, path, body=None, timeout=0):
        calls.append((method, path))
        return {"id": "j1", "status": "IN_QUEUE"} if path == "/run" else next(statuses)

    monkeypatch.setattr(cl, "_call", fake_call)
    monkeypatch.setattr(c.time, "sleep", lambda s: None)
    out = cl.run({"image": "x", "mask": "y", "prompt": "p"}, quiet=True)
    assert out["seed"] == 1
    assert calls[0] == ("POST", "/run") and calls[1] == ("GET", "/status/j1")


def test_failed_job_raises():
    with pytest.raises(c.ClientError, match="FAILED"):
        c.result_from_status({"id": "j", "status": "FAILED", "error": "boom"})
    with pytest.raises(c.ClientError, match="worker error"):
        c.result_from_status({"id": "j", "status": "COMPLETED", "output": {"error": "bad"}})


def test_client_requires_credentials(monkeypatch):
    monkeypatch.setattr(c, "load_config", lambda path=None: {})
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    with pytest.raises(c.ClientError):
        c.SpriteInpaintClient()


def test_load_config_parses_and_env_wins(tmp_path, monkeypatch):
    cfg = tmp_path / "env"
    cfg.write_text("# comment\nRUNPOD_API_KEY='k123'\nRUNPOD_ENDPOINT_ID=ep9\n")
    assert c.load_config(cfg) == {"RUNPOD_API_KEY": "k123", "RUNPOD_ENDPOINT_ID": "ep9"}
    orig = c.load_config
    monkeypatch.setattr(c, "load_config", lambda path=None: orig(cfg))
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "from-env")
    cl = c.SpriteInpaintClient()
    assert cl.api_key == "k123" and cl.endpoint_id == "from-env"


def test_bench_reports_timings(files, monkeypatch):
    sprite, mask, sp, mp, tmp = files
    base = default_input(image=png_b64(sprite, "RGBA"), mask=png_b64(mask, "L"), prompt="p", seed=1)
    out = pipeline.run_job("b", base, fake_runner())
    cl = c.SpriteInpaintClient("ep", "key")
    submitted = []
    monkeypatch.setattr(cl, "submit", lambda inp: submitted.append(inp) or {"id": f"j{len(submitted)}"})
    monkeypatch.setattr(cl, "wait", lambda job_id, timeout=0: {
        "id": job_id, "status": "COMPLETED", "delayTime": 61000, "executionTime": 9500, "output": out})
    rows = c.bench(cl, {"image": "x", "mask": "y", "prompt": "p"}, str(tmp / "b"), str(sp))
    assert [r["delay_s"] for r in rows] == [61.0, 61.0] and rows[0]["execution_s"] == 9.5
    assert {s["seed"] for s in submitted} == {1000, 1001}
    assert (tmp / "b1.png").exists() and (tmp / "b2.png").exists()
