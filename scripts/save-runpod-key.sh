#!/usr/bin/env bash
# Asks for the RunPod API key and endpoint ID without echoing the key, and
# stores them in ~/.config/refactor-sprite-inpaint/env (mode 600) for the client.
set -euo pipefail
dir="$HOME/.config/refactor-sprite-inpaint"
mkdir -p "$dir" && chmod 700 "$dir"
read -rsp "RunPod API key (input hidden): " key; echo
read -rp "Endpoint ID (Overview tab of the endpoint): " ep
umask 077
printf 'RUNPOD_API_KEY=%s\nRUNPOD_ENDPOINT_ID=%s\n' "$key" "$ep" > "$dir/env"
chmod 600 "$dir/env"
echo "Saved to $dir/env (readable only by you)."
