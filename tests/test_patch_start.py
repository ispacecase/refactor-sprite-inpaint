"""scripts/patch-start.sh against the launch lines of worker-comfyui 5.10.0's start.sh."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "patch-start.sh"

# Verbatim from runpod-workers/worker-comfyui 5.10.0 src/start.sh (lines 72 and 78).
UPSTREAM = '''if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"
fi
'''


def run(path):
    return subprocess.run(["bash", str(SCRIPT), str(path)], capture_output=True, text=True)


def test_patches_both_launch_lines(tmp_path):
    f = tmp_path / "start.sh"
    f.write_text(UPSTREAM)
    r = run(f)
    assert r.returncode == 0, r.stderr
    text = f.read_text()
    assert text.count('--log-stdout ${COMFY_EXTRA_ARGS:-} &') == 2
    # The patched line expands the variable into separate arguments.
    line = [l for l in text.splitlines() if "--verbose" in l and "--listen" not in l][0].strip()
    cmd = line.replace("python -u", "printf '%s\\n'").rstrip("&").strip()
    env = dict(os.environ, COMFY_EXTRA_ARGS="--reserve-vram 1.0", COMFY_LOG_LEVEL="INFO")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env).stdout.split()
    assert out[-3:] == ["--log-stdout", "--reserve-vram", "1.0"]
    env.pop("COMFY_EXTRA_ARGS")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env).stdout.split()
    assert out[-1] == "--log-stdout"


def test_fails_if_upstream_changed(tmp_path):
    f = tmp_path / "start.sh"
    f.write_text("python -u /comfyui/main.py --something-else &\n")
    assert run(f).returncode != 0
