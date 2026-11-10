# refactor-sprite-inpaint

A RunPod Serverless worker for **precision repair of Refactor Reality sprites**: fixing a limb, a joint or
an attachment point on a high-resolution glossy neon PNG without touching the rest of the approved art.

It is built on the official [runpod-workers/worker-comfyui](https://github.com/runpod-workers/worker-comfyui)
image and deploys from GitHub through RunPod's GitHub integration (Serverless, not a GPU Pod).

**Status:** the repository is ready to deploy. It has **not** been built on RunPod or run on a GPU yet. The
CPU parts (input validation, crop planning, workflow building, alpha recovery, byte-exact compositing,
client, start-script patch) are covered by 71 unit tests and a build-time self-test.

---

## What a job does

1. Validates the input against the schema (`schemas/input.schema.json`).
2. Crops the sprite around the mask (plus context), flattens transparency onto a matte colour (default
   black), and resizes the crop to about 1 megapixel in multiples of 16.
3. Runs a ComfyUI workflow: **Z-Image-Turbo** + **Fun ControlNet Union 2.1** in inpaint mode, optionally
   with a pose/geometry guide in the same pass. Only the masked latent is denoised.
4. Scales the generation back to the crop's size, recovers alpha (see [Transparency](#transparency)),
   and composites it over the original in premultiplied space with an **inward-only** feather.
5. Copies every pixel whose mask value is 0 **byte-for-byte** from the original and checks that this is
   true before returning (`unmasked_pixels_changed` is always 0, otherwise the job fails).
6. Returns an RGBA PNG (whole image, or just the crop with `return_format: "patch"`), keeping the
   original ICC profile and storing the job settings in an iTXt chunk.

## Model choice

| File (pinned commit, SHA-256 checked) | Size | Licence |
|---|---|---|
| `Comfy-Org/Z-Image-Turbo` → `z_image_turbo_bf16.safetensors` (diffusion model) | 12.3 GB | Apache-2.0 |
| `Comfy-Org/Z-Image-Turbo` → `qwen_3_4b.safetensors` (text encoder) | 8.0 GB | Apache-2.0 |
| `Comfy-Org/Z-Image-Turbo` → `ae.safetensors` (VAE) | 0.3 GB | Apache-2.0 |
| `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` → `…-2.1-2602-8steps.safetensors` | 6.7 GB | Apache-2.0 |

Why this stack:

- **Licence:** all Apache-2.0 and ungated (no Hugging Face token needed), so the outputs can go into a
  commercial game. FLUX.1 Fill [dev], the best-known inpainting model, is under a non-commercial licence.
- **Inpainting and pose in one pass:** the Fun ControlNet Union 2.1 supports an inpaint mode and
  Canny / Depth / Pose / MLSD / HED / Scribble / Gray guides together. ComfyUI 0.34.0 runs it with the
  core `ZImageFunControlnet` node, so **no custom nodes** are installed. The official worker-comfyui
  project ships Z-Image-Turbo with the same ControlNet family, so the pairing is known to work.
- **Fits 24 GB:** see [GPU memory](#gpu-memory-24-gb).
- **Fast iteration:** the 8-step distilled ControlNet runs at 8 steps with cfg 1.

The model choice was not tested on a GPU. Whether it can paint good neon limbs on these sprites will only
be known after the first real jobs (see [First paid test](#first-paid-test)).

## Input

Send `{"input": {...}}`. Full field list with ranges: [`schemas/input.schema.json`](schemas/input.schema.json)
(generated from `src/sprite_inpaint/schema.py`).

| Field | Default | Notes |
|---|---|---|
| `image` | required | 8-bit PNG (RGBA or RGB), base64 / data URI / `https://` URL. 16-bit is rejected. |
| `mask` | required | Same size. Strength is `max(R,G,B) × alpha`, so white on black and white on transparent both work. 0 = untouched. |
| `prompt` | required | What the masked area should become. |
| `guide_image` | null | Pose skeleton (OpenPose colours), scribble, line art, depth or gray map, drawn in the sprite's coordinates. |
| `guide_preprocess` | `none` | `canny` runs ComfyUI's Canny on the guide first (for a rough paint-over). |
| `control_strength` | 0.9 | Model card: 0.65–1.0. Higher keeps inpaint continuity. |
| `seed` | -1 | -1 = random; the seed used is returned. |
| `steps`, `cfg`, `sampler_name`, `scheduler`, `shift`, `denoise` | 8, 1.0, res_multistep, simple, 3.0, 1.0 | Same settings as ComfyUI's official Z-Image-Turbo template. `denoise` < 1 keeps more of the original inside the mask. |
| `work_megapixels`, `max_upscale` | 1.0, 4.0 | Generation size of the crop (max 2.0 MP). |
| `diffusion_precision` | `bf16` | `fp8` halves the diffusion model's VRAM (see [GPU memory](#gpu-memory-24-gb)). |
| `context_padding_px`, `context_padding_ratio` | 96, 0.35 | Context around the mask box (the larger of the two wins). |
| `mask_grow_px` | 6 | Widens only the mask the model sees, not what is composited. |
| `feather_px` | 3 | Inward feather; never reaches outside the mask. |
| `matte_color` | `#000000` | Background used to flatten the sprite and key the result. |
| `alpha_image` | null | Known silhouette of the repaired part (alpha channel, or brightness if it has none). |
| `alpha_mode`, `alpha_key_knee` | `auto`, 0.2 | `auto` = `provided` when `alpha_image` is sent, else `key_add`. See below. |
| `return_format` | `full` | `patch` returns only `crop_box` (smaller response). |
| `return_debug` | false | Adds the model-space source, mask, guide and the raw generation. |

The request limit is 10 MB for `/run` and 20 MB for `/runsync`. For larger sprites, host the PNGs and pass
`https://` URLs.

### Output

```json
{
  "image": "<base64 PNG>", "format": "full", "width": 1254, "height": 1254,
  "crop_box": [587, 462, 1042, 854], "work_size": [1072, 928], "seed": 1234,
  "alpha_mode": "provided", "diffusion_precision": "bf16",
  "masked_pixels": 30000, "unmasked_pixels_changed": 0,
  "models": {"diffusion_models": "z_image_turbo_bf16.safetensors", "...": "..."},
  "worker_version": "1.1.0", "timings": {"comfy_s": 7.1, "total_s": 8.0}
}
```

Errors come back as `{"error": "...", "details": [...]}` and RunPod marks the job FAILED.

### Transparency

The model only sees RGB, so the sprite is flattened onto `matte_color` (black by default) before
generation. Alpha for the repainted pixels then comes from one of these modes:

| `alpha_mode` | Alpha of repainted pixels | Use for |
|---|---|---|
| `auto` (**default**) | `provided` if `alpha_image` is sent, otherwise `key_add` | Everything; just send a silhouette when you have one. |
| `provided` | exactly `alpha_image` | **Precise character-part repairs with a known silhouette** (the recommended way). |
| `key_add` | max(original, keyed) | No silhouette: repairs inside the body and extending a limb outward. Cannot erase. |
| `provided_glow` | max(`alpha_image`, keyed) | Your silhouette plus whatever glow the model paints around it. |
| `preserve` | original | Recolouring or detailing without changing the silhouette. |
| `key` | keyed only | Removing a wrong limb (can make pixels transparent). |

**With a silhouette (`provided`).** Draw the corrected part's shape at the sprite's size: opaque where the
part should be, transparent elsewhere, and soft alpha where its edge or glow should fade. Inside the mask the
result takes exactly this alpha. A masked pixel outside the silhouette becomes transparent, so a wrong old
limb under the mask is erased. The colour comes from the generation, un-matted against the matte colour.
Outside the mask nothing changes, whatever the silhouette contains.

**Without a silhouette (`key_add`).** The result is keyed off the matte:
`keyed alpha = clamp(max|pixel − matte| / alpha_key_knee)`, combined with the original alpha by `max`.
Colour is solved so that the result put back on the matte reproduces the generation exactly, which keeps
the soft alpha of neon glows. Caveat from the real Coder art: body pixels sit at alpha ≈ 253 and a quarter
of them have a brightest channel below about 0.3. A limb extended **into empty space** therefore gets dark
parts with keyed alpha below 1 at the default knee (dark glass on a dark background). Send a silhouette, or
lower `alpha_key_knee` (e.g. 0.08) at the cost of faint glows turning more opaque.

**In every mode** the composite weight is the mask, feathered inward by `feather_px`, and blending is done
in premultiplied space. Mask-0 pixels are copied byte-for-byte (including hidden colour under alpha 0) and
the worker checks this before it returns.

## GPU memory (24 GB)

Only a crop around the mask is generated. A limb-sized mask on a 1254×1254 sprite becomes a crop of about
450×400 px, scaled to ~1 MP (`work_megapixels`). The whole sprite goes through the model only when the mask
spans most of it, and is then scaled to 1 MP as well.

| Item | VRAM (bf16) | VRAM (`diffusion_precision: fp8`) |
|---|---|---|
| Z-Image-Turbo diffusion model | 12.3 GB | ~6.2 GB |
| Fun ControlNet Union 2.1 patch | 6.7 GB | 6.7 GB |
| VAE | 0.3 GB | 0.3 GB |
| Qwen3-4B text encoder | 8.0 GB, only while encoding the prompt, then evicted | same |
| Sampling at ~1 MP | a few GB of activations | same |

How it fits:

- **Dynamic VRAM (ComfyUI 0.34 default).** Weights are loaded only as far as VRAM allows; the rest stay in
  system RAM and stream in, which is slower but does not run out of memory. The text encoder runs first and
  is evicted before the ~19 GB diffusion model and ControlNet load.
- **`COMFY_EXTRA_ARGS="--reserve-vram 1.0"`** (set in the Dockerfile) keeps 1 GB free for the VAE decode
  and allocator spikes. The image's `/start.sh` is patched at build time to pass this variable to ComfyUI,
  so you can change it as a RunPod **environment variable** without rebuilding. Examples:
  `--reserve-vram 2.0` (more headroom), `--fp8_e4m3fn-text-enc` (text encoder in fp8),
  `--disable-smart-memory` (always offload between jobs).
- **Per job:** `diffusion_precision: "fp8"` stores the diffusion weights in fp8 (~13 GB of weights in total,
  everything resident on a 24 GB card) at a small quality cost. Lowering `work_megapixels` reduces
  activation memory.

This is configuration, not a measurement: the worker has not run on a GPU yet. The first test job should
log ComfyUI's model loading (`COMFY_LOG_LEVEL=DEBUG` shows whether weights were partially loaded).

## Client

`client/sprite_inpaint_client.py` uses only the standard library. Pillow and numpy are needed only for
`patch` results and `verify`.

```bash
export RUNPOD_API_KEY=...            # RunPod console > Settings > API Keys
export RUNPOD_ENDPOINT_ID=...        # shown on the endpoint page
```

Build and check a request without sending it (free):

```bash
python client/sprite_inpaint_client.py build-request --image coder.png --mask arm_mask.png --prompt "glossy neon magenta forearm attached at the elbow" --out request.json
```

Submit, wait and save (uses GPU time). Writes `coder_fixed.png` plus a `coder_fixed.json` sidecar:

```bash
python client/sprite_inpaint_client.py submit --image coder.png --mask arm_mask.png --guide arm_pose.png --prompt "glossy neon magenta forearm attached at the elbow" --seed 1234 --out coder_fixed.png
```

Check independently that nothing outside the mask changed (free, local):

```bash
python client/sprite_inpaint_client.py verify --original coder.png --mask arm_mask.png --result coder_fixed.png
```

Other commands: `health`, `status <job>`, `cancel <job>`, `fetch <job> --out x.png [--original coder.png]`,
`submit --no-wait`, `--set key=value` for any schema field, `--debug` to save the model inputs and raw output.
As a library: `SpriteInpaintClient().run(build_input(...))`.

## Deploying on RunPod (GitHub integration)

Nothing here has been deployed. When you are ready:

1. **Push this repo to GitHub** (a private repo is fine).
2. **Connect GitHub:** RunPod console → **Settings** → **Connections** → **GitHub** → **Connect**. Choose
   **Only select repositories** and pick this repo → **Save**.
3. **Create the endpoint:** **Serverless** → **New Endpoint** → under **Import Git Repository** pick this
   repo.
   - **Branch:** `main`
   - **Dockerfile Path:** `Dockerfile` (repo root; build context is the root)
   - **Next**
4. **Endpoint settings:**
   - **Endpoint Name:** `refactor-sprite-inpaint`
   - **Endpoint Type:** **Queue**
   - **GPU Configuration** (priority order): 1st **24 GB — L4, A5000, 3090**; 2nd **24 GB — 4090 PRO**.
     Both have enough VRAM and bf16; the 4090 is faster per job and costs more per second. With fewer than
     five workers, RunPod always uses the highest-priority type that is available.
   - **Active Workers:** `0`
   - **Max Workers:** `1`
   - **GPUs per Worker:** `1`
   - **Idle Timeout:** `5` s (default). Raise it to 60–120 s during an editing session to avoid repeated
     cold starts. The worker is billed while it waits.
   - **FlashBoot:** enabled (default)
   - **Advanced → Execution Timeout:** `600` s (default)
   - **Advanced → CUDA version:** tick **12.8 and every newer version**. The image's torch is `cu128` and
     needs host driver ≥ 570.
   - **Advanced → Data centers:** leave all selected. The models are inside the image, so **no network
     volume**.
   - **Container Disk** (only if the form shows it): **100 GB**. Jobs write only a few MB (uploads and
     outputs are deleted after each job), but RunPod's docs say images are cached on the container disk.
     100 GB covers the whole image (worst case 77.8 GB) either way, and container disk cost is included in
     the worker price.
   - **Environment Variables:** none required. Do **not** set `BUCKET_ENDPOINT_URL` / `BUCKET_*`: this worker
     needs base64 output internally and refuses jobs while they are set.
5. **Deploy Endpoint**, then watch the **Builds** tab.

### Build notes and risks

- **Image size, checked against RunPod's 80 GB limit** (see [Image size](#image-size)): about 56–65 GB
  uncompressed, worst case 77.8 GB.
- RunPod's GitHub builds allow **30 minutes for `docker build`** (160 minutes in total). Downloading 27.4 GB
  is the risk. Each model is its own layer and is SHA-256 checked. If the build hits
  "Build exceeded maximum time limit of 1800 seconds", build the same Dockerfile elsewhere and push it to a
  registry, or move the models to a network volume. Either way the Dockerfile does not change.
- There is no `.runpod/tests.json`, so RunPod's build "Testing" stage starts no paid GPU test jobs.
- Pushing commits does **not** redeploy. Create a **GitHub release** to roll out a new build. Roll back from
  the **Builds** tab.
- The first job after a cold start loads about 27 GB of weights into VRAM, so expect it to be much slower
  than warm jobs. If workers are killed during start-up, raise `RUNPOD_INIT_TIMEOUT` (default 420 s) as an
  environment variable.
- RunPod keeps `/run` results for 30 minutes. `submit` saves the result as soon as the job finishes;
  `fetch <job>` works only inside that window.
- After 7 days with no requests RunPod sets **Max Workers to 0**. Set it back to 1 in the console before
  using the endpoint again.

### Image size

Measured from the registry without pulling the image. Uncompressed sizes come from the gzip size trailer of
each base layer (exact modulo 4 GiB) plus a 300 MB decompression sample of each large layer to pick the
right multiple. Model sizes are the exact byte counts from Hugging Face.

| Part | Size |
|---|---|
| Base `5.10.0-base` (27 layers, compressed) | 14.67 GB |
| Base, uncompressed (most likely) | 29–38 GB (two layers are ambiguous between candidates; samples show 2.35–2.55× compression) |
| Base, uncompressed (worst case, every ambiguous layer at its largest candidate ≤ 4×) | 50.4 GB |
| Models (`z_image_turbo_bf16` 12.31, `qwen_3_4b` 8.04, ControlNet 6.71, VAE 0.34) | 27.40 GB |
| Our code, scripts, start.sh patch | < 1 MB |
| **Total (likely / worst case)** | **56–65 GB / 77.8 GB**, under the 80 GB limit |

Build-time temporary files: each model is downloaded to `<file>.part`, checked and renamed in the same
`RUN`, so no layer holds a second copy. The Dockerfile installs no packages and leaves no caches. The rest of
the builder's disk (base layers, export of the new 27.4 GB of layers) is managed by RunPod.

### First paid test

Free checks first: `health` (shows workers and queue) and `build-request`. Then run one small job with
`--debug` and look at `*.raw_generation.png` to judge the model before tuning prompts and strengths.

## Updating models or ComfyUI

- **Base image:** change the `FROM` tag **and** digest together (Docker Hub:
  `runpod/worker-comfyui`, `<version>-base`). Check that the new ComfyUI still has `ZImageFunControlnet`.
- **Models:** pin a new Hugging Face commit, put its SHA-256 (the `X-Linked-Etag` header of the
  `resolve/<commit>/<file>` URL) in the Dockerfile, and update `MODEL_FILES` in
  `src/sprite_inpaint/workflow.py`. `tests/test_workflow.py` checks that the two agree.
- **Schema:** edit `src/sprite_inpaint/schema.py`, then regenerate the JSON file:

```bash
PYTHONPATH=src python -m sprite_inpaint.schema > schemas/input.schema.json
```

## Development

```bash
pip install numpy pillow scipy runpod pytest
```

```bash
python -m pytest -q tests
```

The workflow template `src/sprite_inpaint/workflows/sprite_inpaint_api.json` is in ComfyUI API format and
can be loaded into a local ComfyUI to experiment. The handler prunes the optional guide, Canny and
negative-prompt nodes per job.

Layout: `Dockerfile`, `scripts/fetch-model.sh` (pinned, checksummed downloads), `src/handler.py` (RunPod
entrypoint wrapping the official handler), `src/sprite_inpaint/` (schema, imaging, workflow, pipeline,
self-test), `client/`, `schemas/`, `tests/`, `DECISIONS.md`.
