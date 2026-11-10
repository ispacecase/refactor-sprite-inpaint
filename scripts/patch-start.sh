#!/usr/bin/env bash
# patch-start [start.sh]
# Appends ${COMFY_EXTRA_ARGS} to both ComfyUI launch lines in worker-comfyui's
# /start.sh, so ComfyUI memory flags (e.g. --reserve-vram) can be set from the
# Dockerfile or RunPod environment variables without forking start.sh.
# Fails the build if the upstream script no longer has the expected lines.
set -euo pipefail

f="${1:-/start.sh}"
sed -i 's|--log-stdout &$|--log-stdout ${COMFY_EXTRA_ARGS:-} \&|' "$f"
n=$(grep -c 'comfyui/main\.py.*--log-stdout \${COMFY_EXTRA_ARGS:-} &$' "$f" || true)
if [ "$n" -ne 2 ]; then
    echo "patch-start: expected to patch 2 ComfyUI launch lines in $f, patched $n" >&2
    exit 1
fi
echo "patch-start: ComfyUI launch lines now take \$COMFY_EXTRA_ARGS"
