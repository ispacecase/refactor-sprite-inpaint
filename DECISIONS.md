# Decisions log

The owner decides; the agent proposes. Each entry says who decided and why. Add new entries at the top.

## 2026-11-10: owner approval and pre-push adjustments (decided by the owner)

The owner approved the custom RunPod Serverless architecture and the Z-Image-Turbo + Fun ControlNet Union
2.1 model selection, and asked for these changes before pushing (to a **public** GitHub repo):

| Decision | Choice |
|---|---|
| 24 GB operation | Keep cropping around the mask (never the whole sprite unless the mask spans it). Rely on ComfyUI's dynamic VRAM; the image passes `COMFY_EXTRA_ARGS` (default `--reserve-vram 1.0`) to ComfyUI through a build-time patch of `/start.sh`. Per-job `diffusion_precision: fp8` halves the diffusion weights. `work_megapixels` capped at 2.0. |
| Transparency | `alpha_mode` default is now `auto`: `provided` whenever an `alpha_image` silhouette is sent (the owner's choice for precise character-part repairs), otherwise `key_add`. All other modes stay available. |
| Image size | Measured from the registry: 56–65 GB likely, 77.8 GB worst case, under RunPod's 80 GB limit. Container disk recommendation raised to 100 GB (cost is included in the worker price). |
| Guarantees | CPU tests and the byte-identical unmasked-pixel check are unchanged; tests grew from 64 to 71. |

## 2026-11-10: initial worker (proposed by Claude; approved by the owner the same day)

The owner's brief: a separate project, a RunPod Serverless ComfyUI worker on the official worker-comfyui
base, deployed through GitHub; endpoint `refactor-sprite-inpaint` with 0 minimum and 1 maximum workers;
no deploy and no paid jobs yet; no invented model URLs; do not touch the Godot game or the ChatGPT MCP.

| Decision | Choice | Reason | Alternatives rejected |
|---|---|---|---|
| Base image | `runpod/worker-comfyui:5.10.0-base`, pinned by digest | Newest release with Docker Hub images (5.11.0 tag exists on GitHub but its images were never published). ComfyUI 0.34.0, torch 2.11.0+cu128. | Unpinned `latest`; forking the worker repo. |
| Model | Z-Image-Turbo bf16 + Fun ControlNet Union 2.1 (2602, 8-step) | Apache-2.0 and ungated, so usable in a commercial game. One core ComfyUI node does inpaint and pose/geometry guidance together. Fits 24 GB. | FLUX.1 Fill [dev] (non-commercial licence, gated); Qwen-Image + InstantX inpaint/union ControlNets (20 GB fp8 model plus two ControlNets is tight on 24 GB); SDXL inpainting (weaker). |
| Custom nodes | None | Every node needed is core in ComfyUI 0.34.0, which leaves fewer build-time failure points. | comfyui_controlnet_aux for automatic pose extraction: it downloads detector weights at runtime, and pose detectors are unreliable on stylised chibi sprites. The caller draws the guide instead. |
| Compositing | In Python (numpy/scipy), after ComfyUI | Byte-exact copies of unmasked pixels are guaranteed and checked; premultiplied blending; inward-only feather. ComfyUI image nodes work on float RGB and drop alpha. | ImageCompositeMasked inside the workflow. |
| Transparency | Flatten on a matte (default black), key the result back with a soft knee, five alpha modes | Keeps partial-alpha glows. Explicit modes for "extend limb" (key_add), "recolour" (preserve), "erase" (key) and "exact silhouette" (provided). | Matting models (glows get thresholded away; BiRefNet/RMBG licences vary). |
| Resolution | Crop around the mask with context and generate at ~1 MP | High-resolution sprites stay full size; the model works at its native scale. | Generating the whole sprite (slow, and the model has to repaint context). |
| Input validation | RunPod `rp_validator` plus our own constraint pass | rp_validator silently skips `constraints` on fields that have defaults (found by tests). | rp_validator alone. |
| Output | Base64 PNG, `full` or `patch` | Fits the response; `patch` keeps big sprites small. S3 upload (`BUCKET_*`) is refused because the wrapper needs the raw generation. | S3 output. |
| Tests during RunPod build | No `.runpod/tests.json` | Avoids paid GPU test jobs at build time; a CPU self-test runs during `docker build` instead. | RunPod Hub test runs. |

Still open: GPU tier priority (cheaper 24 GB first, 4090 second); revisit after the first real jobs.
