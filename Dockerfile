# refactor-sprite-inpaint: RunPod Serverless worker for precision sprite repair.
#
# Base: the official runpod-workers/worker-comfyui image, pinned by tag AND
# digest. 5.10.0 is the newest release with published images (5.11.0 has no
# Docker Hub images); it ships ComfyUI 0.34.0 and torch 2.11.0+cu128
# (needs host driver >= 570, i.e. CUDA 12.8+).
FROM runpod/worker-comfyui:5.10.0-base@sha256:e9d18d3db15839ebb10c10c824109fc2956f31ef7bf21916220c008855b0d538

# ---------------------------------------------------------------------------
# Custom nodes: none. Every node in the workflow (ZImageFunControlnet,
# ModelPatchLoader, Canny, LoadImageMask, SetLatentNoiseMask, ...) is core
# ComfyUI 0.34.0. Add any future ones here with:
#   RUN comfy-node-install <registry-name>
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Models (~27.4 GB). All Apache-2.0 and ungated, so no Hugging Face token is
# needed. Each URL is pinned to a commit and checked against its SHA-256.
# One file per layer so a failed build resumes from the last finished file.
# ---------------------------------------------------------------------------
COPY scripts/fetch-model.sh /usr/local/bin/fetch-model
RUN chmod +x /usr/local/bin/fetch-model

# Z-Image-Turbo diffusion model, bf16 (12.3 GB) - Comfy-Org repack of Tongyi-MAI/Z-Image-Turbo
RUN fetch-model \
    https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
    /comfyui/models/diffusion_models/z_image_turbo_bf16.safetensors \
    2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6

# Qwen3-4B text encoder (8.0 GB)
RUN fetch-model \
    https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/text_encoders/qwen_3_4b.safetensors \
    /comfyui/models/text_encoders/qwen_3_4b.safetensors \
    6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a

# VAE (0.3 GB)
RUN fetch-model \
    https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/vae/ae.safetensors \
    /comfyui/models/vae/ae.safetensors \
    afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38

# alibaba-pai Fun ControlNet Union 2.1, 2602 8-step (6.7 GB): inpaint mode +
# Canny/Depth/Pose/MLSD/HED/Scribble/Gray guides in a single model patch.
RUN fetch-model \
    https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/5155fc56d17821007d6f62ac192c09e0f0e72016/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
    /comfyui/models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
    d1251cc7bc3486bc61d25c3be498ef394c31c85ddf4ee9137d2e933411f4a689

# ---------------------------------------------------------------------------
# GPU memory (24 GB target). ComfyUI 0.34 uses dynamic VRAM by default: weights
# that do not fit stay in system RAM and stream in, and the 8 GB text encoder
# is evicted before sampling. --reserve-vram keeps 1 GB of headroom for the
# VAE decode and allocator spikes. Override COMFY_EXTRA_ARGS in the RunPod
# endpoint's environment variables (e.g. "--reserve-vram 2.0") without a rebuild.
# ---------------------------------------------------------------------------
COPY scripts/patch-start.sh /usr/local/bin/patch-start
RUN chmod +x /usr/local/bin/patch-start && patch-start /start.sh
ENV COMFY_EXTRA_ARGS="--reserve-vram 1.0"

# ---------------------------------------------------------------------------
# Handler: keep the official one as a library, put ours in front of it.
# ---------------------------------------------------------------------------
RUN mv /handler.py /comfy_worker_handler.py
COPY src/sprite_inpaint /sprite_inpaint
COPY src/handler.py /handler.py
# Replaces the upstream example; only used if the handler is started locally
# without RunPod's job queue (e.g. `python /handler.py` in a dev container).
COPY test_input.json /test_input.json

# Build-time check (CPU only): template <-> model files, graph pruning and
# byte-exact compositing. Fails the build instead of a live worker.
RUN cd / && SELFTEST_MODELS_DIR=/comfyui/models python -m sprite_inpaint.selftest

ENV COMFY_LOG_LEVEL=INFO

# CMD is inherited: /start.sh launches ComfyUI, then `python -u /handler.py`.
