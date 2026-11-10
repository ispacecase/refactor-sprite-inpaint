#!/usr/bin/env python3
"""Client for the refactor-sprite-inpaint RunPod Serverless endpoint.

Standard library only for normal use. Pillow + numpy are needed only for
``--return-format patch`` reassembly and the ``verify`` command.

Environment:
    RUNPOD_API_KEY       RunPod API key (Settings > API Keys)
    RUNPOD_ENDPOINT_ID   endpoint id shown on the endpoint's page

Examples:
    # Build and check a request without sending anything (free):
    python client/sprite_inpaint_client.py build-request \\
        --image coder.png --mask arm_mask.png --prompt "..." --out request.json

    # Submit (costs GPU time), wait, save the result:
    python client/sprite_inpaint_client.py submit \\
        --image coder.png --mask arm_mask.png --guide arm_pose.png \\
        --prompt "glossy neon magenta forearm attached at the elbow joint" \\
        --set seed=1234 --out coder_fixed.png

    # Independently confirm unmasked pixels are byte-identical:
    python client/sprite_inpaint_client.py verify \\
        --original coder.png --mask arm_mask.png --result coder_fixed.png
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://api.runpod.ai/v2"
RUN_LIMIT = 10 * 1024 * 1024       # RunPod /run request limit
RUNSYNC_LIMIT = 20 * 1024 * 1024   # RunPod /runsync request limit
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "input.schema.json"
CONFIG_PATH = Path(os.environ.get("SPRITE_INPAINT_CONFIG",
                                  Path.home() / ".config" / "refactor-sprite-inpaint" / "env"))


def load_config(path: Path = CONFIG_PATH) -> dict:
    """KEY=VALUE lines (RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID). Env vars win."""
    cfg = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return cfg


class ClientError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Request building and offline validation
# ---------------------------------------------------------------------------

def encode_image(path_or_url: str) -> str:
    """Local file -> base64; https:// URLs are passed through for the worker."""
    if path_or_url.startswith("https://"):
        return path_or_url
    return base64.b64encode(Path(path_or_url).read_bytes()).decode("ascii")


def load_schema(path: Path = SCHEMA_PATH) -> dict | None:
    try:
        return json.loads(path.read_text())
    except OSError:
        return None


def validate_input(inp: dict, schema: dict | None) -> list[str]:
    """Small validator for this flat schema (types, ranges, enums, required)."""
    if schema is None:
        return []
    errs = []
    props = schema["properties"]
    for k in inp:
        if k not in props:
            errs.append(f"unknown field '{k}'")
    for k in schema.get("required", []):
        if k not in inp:
            errs.append(f"missing required field '{k}'")
    py = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}
    for k, v in inp.items():
        p = props.get(k)
        if p is None:
            continue
        types = p["type"] if isinstance(p["type"], list) else [p["type"]]
        ok = any(isinstance(v, py[t]) and not (t in ("integer", "number") and isinstance(v, bool))
                 for t in types)
        if not ok:
            errs.append(f"'{k}' should be {'/'.join(types)}, got {type(v).__name__}")
            continue
        if "enum" in p and v not in p["enum"]:
            errs.append(f"'{k}' must be one of {p['enum']}")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "minimum" in p and v < p["minimum"]:
                errs.append(f"'{k}' must be >= {p['minimum']}")
            if "maximum" in p and v > p["maximum"]:
                errs.append(f"'{k}' must be <= {p['maximum']}")
        if isinstance(v, str) and "maxLength" in p and len(v) > p["maxLength"]:
            errs.append(f"'{k}' longer than {p['maxLength']}")
    return errs


def build_input(image: str, mask: str, prompt: str, *, guide: str | None = None,
                alpha_image: str | None = None, **settings) -> dict:
    inp = {"image": encode_image(image), "mask": encode_image(mask), "prompt": prompt}
    if guide:
        inp["guide_image"] = encode_image(guide)
    if alpha_image:
        inp["alpha_image"] = encode_image(alpha_image)
    inp.update({k: v for k, v in settings.items() if v is not None})
    errs = validate_input(inp, load_schema())
    if errs:
        raise ClientError("invalid request: " + "; ".join(errs))
    return inp


# ---------------------------------------------------------------------------
# RunPod API
# ---------------------------------------------------------------------------

class SpriteInpaintClient:
    def __init__(self, endpoint_id: str | None = None, api_key: str | None = None,
                 base_url: str = API_BASE):
        cfg = load_config()
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID") or cfg.get("RUNPOD_ENDPOINT_ID")
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY") or cfg.get("RUNPOD_API_KEY")
        if not self.endpoint_id:
            raise ClientError(f"set RUNPOD_ENDPOINT_ID, pass --endpoint-id, or add it to {CONFIG_PATH}")
        if not self.api_key:
            raise ClientError(f"set RUNPOD_API_KEY or add it to {CONFIG_PATH}")
        self.base = f"{base_url}/{self.endpoint_id}"

    def _call(self, method: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise ClientError(f"HTTP {exc.code} from {path}: {detail}") from exc

    def health(self) -> dict:
        return self._call("GET", "/health")

    def submit(self, inp: dict, *, sync: bool = False) -> dict:
        body = {"input": inp}
        size = len(json.dumps(body))
        limit = RUNSYNC_LIMIT if sync else RUN_LIMIT
        if size > limit:
            raise ClientError(
                f"request is {size / 1e6:.1f} MB, over RunPod's {limit / 1e6:.0f} MB limit for "
                f"{'/runsync' if sync else '/run'}; host the PNGs at https:// URLs and pass those instead")
        return self._call("POST", "/runsync" if sync else "/run", body, timeout=330 if sync else 120)

    def status(self, job_id: str) -> dict:
        return self._call("GET", f"/status/{job_id}")

    def cancel(self, job_id: str) -> dict:
        return self._call("POST", f"/cancel/{job_id}")

    def wait(self, job_id: str, timeout: float = 900, quiet: bool = False) -> dict:
        start, delay, last = time.time(), 2.0, None
        while True:
            st = self.status(job_id)
            s = st.get("status")
            if s != last and not quiet:
                print(f"[{time.time() - start:6.1f}s] {job_id}: {s}", file=sys.stderr)
                last = s
            if s in TERMINAL:
                return st
            if time.time() - start > timeout:
                raise ClientError(f"gave up waiting after {timeout}s; job {job_id} is still {s} "
                                  f"(it keeps running - cancel it with: cancel {job_id})")
            time.sleep(delay)
            delay = min(delay * 1.5, 10.0)

    def run(self, inp: dict, *, timeout: float = 900, quiet: bool = False) -> dict:
        job = self.submit(inp)
        if not quiet:
            print(f"submitted job {job['id']}", file=sys.stderr)
        st = self.wait(job["id"], timeout=timeout, quiet=quiet)
        return result_from_status(st)


def bench(client: "SpriteInpaintClient", inp: dict, out_prefix: str, original: str,
          timeout: float = 1800) -> list[dict]:
    """Submit two identical jobs at once (max 1 worker): job 1 pays the cold
    start, job 2 queues behind it and runs warm. Returns RunPod timings."""
    t0 = time.time()
    ids = [client.submit(dict(inp, seed=1000 + i))["id"] for i in range(2)]
    rows = []
    for n, job_id in enumerate(ids, 1):
        st = client.wait(job_id, timeout=timeout)
        out = result_from_status(st)
        save_result(out, f"{out_prefix}{n}.png", original)
        rows.append({
            "job": n,
            "id": job_id,
            "delay_s": round((st.get("delayTime") or 0) / 1000, 1),
            "execution_s": round((st.get("executionTime") or 0) / 1000, 1),
            "comfy_s": out["timings"].get("comfy_s"),
            "wall_s_since_submit": round(time.time() - t0, 1),
            "unmasked_pixels_changed": out["unmasked_pixels_changed"],
        })
    return rows


def result_from_status(st: dict) -> dict:
    s = st.get("status")
    out = st.get("output")
    if s != "COMPLETED":
        raise ClientError(f"job {st.get('id')} ended {s}: {st.get('error') or out}")
    if not isinstance(out, dict):
        raise ClientError(f"unexpected output: {str(out)[:500]}")
    if "error" in out:
        raise ClientError(f"worker error: {out['error']} {out.get('details') or ''}".strip())
    return out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def result_png(out: dict, original_path: str | None) -> bytes:
    png = base64.b64decode(out["image"])
    if out.get("format") != "patch":
        return png
    if not original_path or original_path.startswith("https://"):
        raise ClientError("patch results need the local original image to paste into")
    import io

    from PIL import Image  # optional dependency

    base = Image.open(original_path)
    base.load()
    icc = base.info.get("icc_profile")
    base = base.convert("RGBA")
    patch = Image.open(io.BytesIO(png)).convert("RGBA")
    x0, y0, x1, y1 = out["crop_box"]
    if patch.size != (x1 - x0, y1 - y0):
        raise ClientError("patch size does not match crop_box")
    base.paste(patch, (x0, y0))  # replace, not alpha-blend: the patch already holds final pixels
    buf = io.BytesIO()
    base.save(buf, "PNG", **({"icc_profile": icc} if icc else {}))
    return buf.getvalue()


def save_result(out: dict, dest: str, original_path: str | None) -> None:
    Path(dest).write_bytes(result_png(out, original_path))
    meta = {k: v for k, v in out.items() if k not in ("image", "debug")}
    Path(dest).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    dbg = out.get("debug")
    if dbg:
        stem = Path(dest).with_suffix("")
        for k, v in dbg.items():
            if v:
                Path(f"{stem}.{k}.png").write_bytes(base64.b64decode(v))


def verify(original: str, mask: str, result: str) -> tuple[int, int]:
    """(unmasked pixels that differ, masked pixel count)."""
    import numpy as np
    from PIL import Image

    o = np.array(Image.open(original).convert("RGBA"))
    r = np.array(Image.open(result).convert("RGBA"))
    m = np.asarray(Image.open(mask).convert("RGBA"), dtype=np.float32) / 255.0
    sel = (m[..., :3].max(-1) * m[..., 3]) > 0
    if o.shape != r.shape:
        raise ClientError(f"size mismatch {o.shape} vs {r.shape}")
    diff = np.any(o != r, axis=-1)
    return int((diff & ~sel).sum()), int(sel.sum())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_sets(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ClientError(f"--set expects key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _request_from_args(a) -> dict:
    settings = _parse_sets(a.set)
    for k in ("seed", "steps", "denoise", "control_strength", "alpha_mode", "guide_preprocess",
              "return_format", "diffusion_precision"):
        v = getattr(a, k, None)
        if v is not None:
            settings[k] = v
    if a.debug:
        settings["return_debug"] = True
    return build_input(a.image, a.mask, a.prompt, guide=a.guide, alpha_image=a.alpha_image, **settings)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint-id")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def job_args(p):
        p.add_argument("--image", required=True, help="sprite PNG path or https:// URL")
        p.add_argument("--mask", required=True, help="mask path/URL (white = repaint)")
        p.add_argument("--prompt", required=True)
        p.add_argument("--guide", help="pose/geometry guide path/URL")
        p.add_argument("--alpha-image", help="known silhouette of the repaired part (enables alpha_mode provided)")
        p.add_argument("--seed", type=int)
        p.add_argument("--steps", type=int)
        p.add_argument("--denoise", type=float)
        p.add_argument("--control-strength", type=float)
        p.add_argument("--alpha-mode", help="default auto: 'provided' if --alpha-image is given, else 'key_add'")
        p.add_argument("--precision", dest="diffusion_precision", choices=["bf16", "fp8"],
                       help="diffusion weights in VRAM (default bf16)")
        p.add_argument("--guide-preprocess")
        p.add_argument("--return-format", choices=["full", "patch"])
        p.add_argument("--debug", action="store_true", help="also save model inputs and raw output")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="any other schema field (JSON value)")

    b = sub.add_parser("build-request", help="write the request JSON; sends nothing")
    job_args(b)
    b.add_argument("--out", required=True)

    s = sub.add_parser("submit", help="submit a job, wait, and save the PNG (uses GPU time)")
    job_args(s)
    s.add_argument("--out", required=True, help="output PNG (a .json sidecar is written too)")
    s.add_argument("--timeout", type=float, default=900)
    s.add_argument("--no-wait", action="store_true", help="print the job id and exit")

    for name in ("status", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("job_id")
    f = sub.add_parser("fetch", help="save the result of a finished job")
    f.add_argument("job_id")
    f.add_argument("--out", required=True)
    f.add_argument("--original", help="original PNG (needed for patch results)")
    sub.add_parser("health")

    bn = sub.add_parser("bench", help="2 jobs: cold start + warm repair timings (uses GPU time)")
    job_args(bn)
    bn.add_argument("--out-prefix", required=True, help="saves <prefix>1.png and <prefix>2.png")
    bn.add_argument("--timeout", type=float, default=1800)

    v = sub.add_parser("verify", help="check unmasked pixels are byte-identical (local, free)")
    v.add_argument("--original", required=True)
    v.add_argument("--mask", required=True)
    v.add_argument("--result", required=True)

    a = ap.parse_args(argv)
    try:
        if a.cmd == "build-request":
            inp = _request_from_args(a)
            Path(a.out).write_text(json.dumps({"input": inp}))
            print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB); nothing was sent")
            return 0
        if a.cmd == "verify":
            changed, masked = verify(a.original, a.mask, a.result)
            print(f"masked pixels: {masked}; unmasked pixels changed: {changed}")
            return 0 if changed == 0 else 1

        client = SpriteInpaintClient(a.endpoint_id)
        if a.cmd == "health":
            print(json.dumps(client.health(), indent=2))
        elif a.cmd == "status":
            st = client.status(a.job_id)
            if isinstance(st.get("output"), dict):
                st["output"] = {k: ("<base64>" if k in ("image", "debug") else v) for k, v in st["output"].items()}
            print(json.dumps(st, indent=2))
        elif a.cmd == "cancel":
            print(json.dumps(client.cancel(a.job_id), indent=2))
        elif a.cmd == "fetch":
            out = result_from_status(client.status(a.job_id))
            save_result(out, a.out, a.original)
            print(f"saved {a.out}")
        elif a.cmd == "bench":
            rows = bench(client, _request_from_args(a), a.out_prefix, a.image, a.timeout)
            for r in rows:
                print(json.dumps(r))
            print("job 1: delay_s = queue + cold start (container start, ComfyUI boot); "
                  "execution_s = model load + repair. job 2: execution_s = warm repair.")
        elif a.cmd == "submit":
            inp = _request_from_args(a)
            job = client.submit(inp)
            print(f"submitted job {job['id']}", file=sys.stderr)
            if a.no_wait:
                print(job["id"])
                return 0
            out = result_from_status(client.wait(job["id"], timeout=a.timeout))
            save_result(out, a.out, a.image)
            print(f"saved {a.out}  seed={out['seed']}  crop={out['crop_box']}  "
                  f"unmasked_changed={out['unmasked_pixels_changed']}  timings={out['timings']}")
        return 0
    except ClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
