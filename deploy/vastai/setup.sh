#!/usr/bin/env bash
#
# Provision a rented Vast.ai GPU instance to serve muscriptor's note
# transcription API. Runs *on the instance*; `provision.sh` is the wrapper that
# copies the tree up and calls this over SSH.
#
# Why native and not the repo's Docker image: a Vast.ai instance is itself an
# unprivileged container. There is no Docker-in-Docker, so `Dockerfile` /
# `swarm.yml` / `deploy.sh` (the Kyutai Swarm path) cannot be reused here. This
# script reproduces what that image does — uv venv, HF cache prewarm, one
# long-lived `muscriptor serve` — using the image's own supervisor + Caddy
# conventions instead. Both deployment paths stay valid; neither replaces the
# other.
#
# Re-running is safe. Every step is written to converge rather than append, and
# the two expensive ones (dependency install, weight download) no-op once their
# artifacts are in place. See "Idempotency" in README.md for the exceptions.

set -euo pipefail

APP_DIR="${MUSCRIPTOR_APP_DIR:-/workspace/muscriptor}"
ENV_FILE="${WORKSPACE:-/workspace}/.env"
INTERNAL_PORT="${MUSCRIPTOR_INTERNAL_PORT:-18000}"
MODEL_SIZE="${MUSCRIPTOR_MODEL:-medium}"
PORTAL_LABEL="MuScriptor"

log() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 0. Sanity: are we actually on a Vast instance with a GPU?
# ---------------------------------------------------------------------------
log "Checking the environment"
command -v uv >/dev/null || { echo "uv not found on PATH" >&2; exit 1; }
command -v supervisorctl >/dev/null || {
    echo "supervisorctl not found — this does not look like a Vast base image" >&2
    exit 1
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# ---------------------------------------------------------------------------
# 1. HF_TOKEN
#
# The default weights (hf://MuScriptor/muscriptor-<size>) are a `gated: auto`
# repo: public metadata, but the file download 401s without an authenticated
# token that has accepted the licence. The token lives ONLY in ${WORKSPACE}/.env
# (mode 600), which the image sources into login shells and supervisor services.
# It is never written into the repo, this script, or the supervisor config, so
# nothing secret is committable.
# ---------------------------------------------------------------------------
log "Checking HF_TOKEN"
if [[ ! -f "${ENV_FILE}" ]] || ! grep -q '^HF_TOKEN=' "${ENV_FILE}"; then
    cat >&2 <<EOF
HF_TOKEN is not set in ${ENV_FILE}.

The model weights are gated, so the server cannot start without it. From your
own machine (this keeps the token out of your shell history and off argv):

    ssh -p <port> root@<host> 'umask 077; IFS= read -r t; \\
        printf "HF_TOKEN=%s\\n" "\$t" >> ${ENV_FILE}' <<< 'hf_xxxxxxxx'

Then re-run this script.
EOF
    exit 1
fi
chmod 600 "${ENV_FILE}"
set -a
# shellcheck source=/dev/null
. "${ENV_FILE}"
set +a
echo "HF_TOKEN present (${#HF_TOKEN} chars)"

# ---------------------------------------------------------------------------
# 2. Python dependencies
#
# `uv sync` honours uv.lock — same as the Docker image — with one deliberate
# exclusion: torch.
#
# uv.lock pins the current PyPI torch, which is a CUDA 13.0 build. Vast hosts
# very commonly run a 12.x driver (this was written against 570.181, a CUDA 12.8
# ceiling), and CUDA *major* forward-compatibility is a datacenter-GPU-only
# feature, so it does not apply to the consumer cards that make up most of the
# marketplace. On such a host the locked wheel imports and then reports
# `torch.cuda.is_available() == False` ("the NVIDIA driver on your system is too
# old") — the server silently serves every request on CPU.
#
# So torch is held back here and installed from the driver-matching
# download.pytorch.org index below. Everything else stays exactly as locked.
#
# Holding it back (rather than syncing and then reinstalling) matters for a
# practical reason: doing it the other way downloads two ~2.5 GB CUDA torch
# builds, and on a 16 GB instance that runs the disk to 100% and dies with
# ENOSPC partway through. One download, one build.
# ---------------------------------------------------------------------------
# torchaudio is held back for the same reason, and must come from a compatible
# release on the *same CUDA index* as torch. A wheel for a different runtime
# variant can install cleanly and then die while loading its extension:
#   OSError: Could not load this library: .../torchaudio/lib/_torchaudio.abi3.so
# beat-this imports it, so the damage shows up as tempo detection failing
# mid-request: /transcribe streams notes and then aborts before
# `transcription_complete`, which is a confusing way to discover an ABI problem.
log "Installing Python dependencies (uv sync, torch/torchaudio held back)"
cd "${APP_DIR}"
# hardlink rather than copy: the uv cache and the venv are on the same
# filesystem here, so hardlinking shares the extents instead of doubling ~7 GB
# of CUDA libraries. (The Dockerfile uses copy because its cache mount is a
# different filesystem.)
export UV_LINK_MODE=hardlink
export UV_CACHE_DIR="${WORKSPACE:-/workspace}/.uv-cache"
# Hold back torch, torchaudio, and the CUDA runtime packages that are in the
# lock only because torch depends on them. Holding back just torch is not
# enough: uv still installs the lock's `nvidia-*-cu13` / `triton` / `cuda-*`
# set, and the cu12 torch below then pulls its own parallel copy. That wastes
# several GB, and — because `uv sync` restores the cu13 set on every run while
# the torch step reinstalls the cu12 set — it makes every re-provision churn
# ~4 GB of downloads and push a 16 GB disk to 90% full.
#
# The list is derived from uv.lock rather than hardcoded so it cannot drift out
# of sync with a lockfile update. Everything held back here is supplied by the
# `uv pip install torch torchaudio` below, which resolves torch's dependency
# tree from the CUDA-matched index.
holdback=()
while IFS= read -r pkg; do
    holdback+=(--no-install-package "${pkg}")
done < <(
    grep -oE '^name = "(torch|torchaudio|triton|pytorch-triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)"' uv.lock \
        | sed -E 's/^name = "//; s/"$//' | sort -u
)
echo "Holding back ${#holdback[@]} package(s) for the CUDA-matched install"
uv sync --no-dev "${holdback[@]}"

log "Installing torch/torchaudio builds that match the host driver"
max_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
echo "Driver supports up to CUDA ${max_cuda}"
case "${max_cuda}" in
    12.[0-5]*) idx=cu121 ;;
    12.[6-7]*) idx=cu126 ;;
    12.*)      idx=cu128 ;;
    13.*)      idx=cu130 ;;
    *)         echo "Unrecognised CUDA version '${max_cuda}'" >&2; exit 1 ;;
esac
echo "Installing torch and torchaudio from the ${idx} index"
# Note the deliberate absence of --extra-index-url: mixing PyPI in makes uv
# resolve torch from the first index carrying the name (PyPI), after which the
# +cuXXX local version looks unsatisfiable.
#
# Installed in one command so uv resolves them together and cannot pair a torch
# with a torchaudio built against a different one.
uv pip install --python .venv/bin/python \
    torch torchaudio --index-url "https://download.pytorch.org/whl/${idx}"

./.venv/bin/python - <<'PY'
import sys

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU — check the driver/wheel CUDA match above")
print("device", torch.cuda.get_device_name(0))

# Import torchaudio explicitly: it loads a compiled extension against the torch
# runtime above, and if that pairing is wrong we want to fail here, loudly,
# rather than halfway through the first /transcribe request.
import torchaudio

print("torchaudio", torchaudio.__version__, "extension loaded")

# beat-this is what actually imports torchaudio in the request path.
from beat_this.inference import Audio2Beats  # noqa: F401

print("beat_this import ok")
PY


# ---------------------------------------------------------------------------
# 3. Warm the model caches
#
# Mirrors the root Dockerfile's soundfont prewarm, so a cold first request
# doesn't stall behind ~1.2 GB of downloads.
#
# Two caches, in two places, which is easy to get wrong:
#   - hf:// URLs land in HF_HOME, ${WORKSPACE}/.hf_home on this image.
#   - plain http(s) URLs land in ~/.cache/muscriptor (see
#     muscriptor/utils/download.py). The BTC chord checkpoint is one of these,
#     so it sits OUTSIDE the workspace and is therefore not covered even by a
#     mounted volume. Fetch it here rather than paying for it on the first
#     request that asks for chords.
# ---------------------------------------------------------------------------
log "Prefetching weights, soundfont and chord checkpoint"
./.venv/bin/python - "${MODEL_SIZE}" <<'PY'
import sys

from muscriptor.soundfonts import SF3_URL
from muscriptor.transcription_model import _HF_REPO_TEMPLATE
from muscriptor.utils.chords import CHECKPOINT_URL
from muscriptor.utils.download import download_if_necessary

size = sys.argv[1]
for url in (_HF_REPO_TEMPLATE.format(size=size), SF3_URL, CHECKPOINT_URL):
    print("fetching", url, flush=True)
    print("  ->", download_if_necessary(url), flush=True)
PY

# ---------------------------------------------------------------------------
# 4. Install the supervisor service
# ---------------------------------------------------------------------------
log "Installing the supervisor service"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install -m 755 "${here}/supervisor/muscriptor.sh" /opt/supervisor-scripts/muscriptor.sh
install -m 644 "${here}/supervisor/muscriptor.conf" /etc/supervisor/conf.d/muscriptor.conf

# A remotely managed Cloudflare Tunnel supplies browser-trusted HTTPS and a
# stable hostname. It is optional: without a token the loopback service and
# token-authenticated Vast Caddy edge still work for private/operator access.
if [[ -s "${WORKSPACE:-/workspace}/.cloudflared-token" ]]; then
    command -v /opt/instance-tools/bin/cloudflared >/dev/null || {
        echo "cloudflared not found at the Vast base-image path" >&2
        exit 1
    }
    install -m 755 "${here}/supervisor/muscriptor-tunnel.sh" \
        /opt/supervisor-scripts/muscriptor-tunnel.sh
    install -m 644 "${here}/supervisor/muscriptor-tunnel.conf" \
        /etc/supervisor/conf.d/muscriptor-tunnel.conf
    echo "Cloudflare named-tunnel connector enabled"
else
    echo "No Cloudflare tunnel token; named-tunnel connector not installed"
fi

# The wrapper reads these; putting them in .env keeps the service definition
# free of instance-specific values. `grep -v` first so re-runs replace rather
# than accumulate duplicate lines.
tmp="$(mktemp)"
grep -vE '^(MUSCRIPTOR_APP_DIR|MUSCRIPTOR_INTERNAL_PORT|MUSCRIPTOR_MODEL)=' \
    "${ENV_FILE}" > "${tmp}" || true
{
    echo "MUSCRIPTOR_APP_DIR=${APP_DIR}"
    echo "MUSCRIPTOR_INTERNAL_PORT=${INTERNAL_PORT}"
    echo "MUSCRIPTOR_MODEL=${MODEL_SIZE}"
} >> "${tmp}"
mv "${tmp}" "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

# ---------------------------------------------------------------------------
# 5. Publish it on a free external port, behind the Caddy auth edge
#
# External ports are allocated when the instance is *created* and cannot be
# added later, so this picks one that is already open and unused rather than
# assuming a number. Caddy only creates an authenticated external vhost when
# external_port != internal_port and VAST_TCP_PORT_<external_port> exists,
# which is why the app listens on 18000 and is published on a separate port.
# ---------------------------------------------------------------------------
log "Publishing through the Caddy edge"
python3 - "${PORTAL_LABEL}" "${INTERNAL_PORT}" <<'PY'
import json
import os
import subprocess
import sys

import yaml

label, internal_port = sys.argv[1], int(sys.argv[2])
caps = json.loads(subprocess.check_output(["vast-capabilities"]))
ports = caps["instance"]["open_ports"]

path = "/etc/portal.yaml"
with open(path) as fh:
    doc = yaml.safe_load(fh) or {}
apps = doc.setdefault("applications", {})

existing = apps.get(label, {}).get("external_port")
candidates = [
    p for p in ports
    if p["proto"] == "tcp"
    and not p.get("self_mapped")
    and p["container_port"] > 1024
    and (not p["in_use"] or p["container_port"] == existing)
]
if not candidates:
    sys.exit(
        "No free normal external port on this instance. External ports are "
        "fixed at creation — re-rent with more ports open (see README.md)."
    )

chosen = existing if existing in [c["container_port"] for c in candidates] \
    else candidates[0]["container_port"]
public = os.environ[f"VAST_TCP_PORT_{chosen}"]

apps[label] = {
    "hostname": "localhost",
    "external_port": chosen,
    "internal_port": internal_port,
    "open_path": "/health",
    "name": label,
}
with open(path, "w") as fh:
    yaml.safe_dump(doc, fh, sort_keys=False)

print(f"portal entry: external {chosen} -> internal {internal_port}")
print(f"public endpoint: http://{os.environ['PUBLIC_IPADDR']}:{public}")
PY

# ---------------------------------------------------------------------------
# 6. Start it
# ---------------------------------------------------------------------------
log "Starting the service"
supervisorctl reread
supervisorctl update
# `restart` covers both cases (it prints a harmless "ERROR (not running)" when
# the service was stopped, then starts it).
supervisorctl restart muscriptor
supervisorctl restart caddy
if [[ -s "${WORKSPACE:-/workspace}/.cloudflared-token" ]]; then
    supervisorctl restart muscriptor_named_tunnel
fi

log "Done. Follow the model load with:"
echo "    tail -f /var/log/portal/muscriptor.log"
echo
echo "Health check once it reports 'Uvicorn running':"
echo "    curl -s localhost:${INTERNAL_PORT}/health"
