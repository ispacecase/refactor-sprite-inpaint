"""RunPod Serverless entrypoint for refactor-sprite-inpaint.

The official worker-comfyui image starts ComfyUI and then runs /handler.py.
Our Dockerfile renames the official handler to /comfy_worker_handler.py and
installs this file as /handler.py. We validate the high-level input, build the
ComfyUI workflow, let the official handler run it (upload, queue, websocket,
history, image fetch), then composite the result exactly in Python.
"""

from __future__ import annotations

import base64
import os
import traceback

import runpod

from sprite_inpaint import pipeline
from sprite_inpaint.imaging import InputError
from sprite_inpaint.schema import validate_input

COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")


class ComfyError(RuntimeError):
    def __init__(self, message: str, details=None):
        super().__init__(message)
        self.details = details


def _upstream():
    import comfy_worker_handler  # the official worker-comfyui handler, renamed

    return comfy_worker_handler


def _cleanup(images: list, result: dict | None) -> None:
    """Remove this job's uploads and outputs so warm workers don't fill disk."""
    paths = [os.path.join(COMFY_DIR, "input", i["name"]) for i in images]
    for img in (result or {}).get("images", []) or []:
        fn = img.get("filename")
        if fn:
            paths.append(os.path.join(COMFY_DIR, "output", "sprite_inpaint", os.path.basename(fn)))
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def make_comfy_runner(job_id: str):
    def run(workflow: dict, images: list) -> bytes:
        if os.environ.get("BUCKET_ENDPOINT_URL"):
            raise ComfyError(
                "BUCKET_ENDPOINT_URL is set, which makes worker-comfyui upload raw output to S3. "
                "This worker composites in-process and needs base64 output: remove the BUCKET_* env vars.")
        result = None
        try:
            result = _upstream().handler({"id": job_id, "input": {"workflow": workflow, "images": images}})
            if "error" in result:
                raise ComfyError(result["error"], result.get("details"))
            outs = [i for i in result.get("images", []) if i.get("type") == "base64"]
            if not outs:
                raise ComfyError("ComfyUI produced no image", result.get("errors"))
            return base64.b64decode(outs[0]["data"])
        finally:
            _cleanup(images, result)

    return run


def handler(job: dict) -> dict:
    job_id = job.get("id", "local")
    values, errors = validate_input(job.get("input") or {})
    if errors:
        return {"error": "Invalid input", "details": errors}
    try:
        return pipeline.run_job(job_id, values, make_comfy_runner(job_id))
    except InputError as exc:
        return {"error": f"Invalid input: {exc}"}
    except ComfyError as exc:
        return {"error": f"ComfyUI failed: {exc}", "details": exc.details}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {"error": f"Unexpected error: {exc}"}


if __name__ == "__main__":
    print("refactor-sprite-inpaint - starting handler")
    runpod.serverless.start({"handler": handler})
