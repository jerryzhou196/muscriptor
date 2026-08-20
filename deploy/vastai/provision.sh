#!/usr/bin/env bash
#
# Copy this checkout onto a rented Vast.ai instance and provision it.
# Runs on YOUR machine; everything it does remotely lives in setup.sh.
#
#   ./deploy/vastai/provision.sh root@1.2.3.4 17762
#
# The SSH host and port come from the Vast console ("Connect" → the
# `ssh -p <port> root@<host>` line). They change every time you rent, which is
# most of why this is a script and not a one-liner in a runbook.

set -euo pipefail

usage() {
    cat >&2 <<EOF
usage: $0 <user@host> <ssh-port>

Example:
    $0 root@203.0.113.10 17762

Set deployment values in your environment to install them on the instance.
The two tokens are sent over stdin, so they never land in argv or shell
history:

    HF_TOKEN=hf_xxxx \
    CF_TUNNEL_TOKEN=eyJ... \
    MUSCRIPTOR_ALLOWED_ORIGINS=https://muscriptor.example.vercel.app \
      $0 root@1.2.3.4 17762
EOF
    exit 2
}

[[ $# -eq 2 ]] || usage
TARGET="$1"
PORT="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_DIR="${MUSCRIPTOR_APP_DIR:-/workspace/muscriptor}"
SSH=(ssh -o BatchMode=yes -p "${PORT}")

echo "==> Syncing $(basename "${REPO_ROOT}") to ${TARGET}:${REMOTE_DIR}"
# Excludes mirror .dockerignore's intent: no VCS metadata, no node_modules, and
# not the local .venv — the instance builds its own for a different platform.
rsync -az --delete \
    -e "ssh -o BatchMode=yes -p ${PORT}" \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'web/node_modules' \
    --exclude 'muscriptor/web_dist' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "${REPO_ROOT}/" "${TARGET}:${REMOTE_DIR}/"

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "==> Installing HF_TOKEN into /workspace/.env"
    "${SSH[@]}" "${TARGET}" \
        'umask 077; IFS= read -r t || true; touch /workspace/.env;
         grep -v "^HF_TOKEN=" /workspace/.env > /workspace/.env.new 2>/dev/null || true;
         printf "HF_TOKEN=%s\n" "$t" >> /workspace/.env.new;
         mv /workspace/.env.new /workspace/.env; chmod 600 /workspace/.env;
         echo "   token installed (${#t} chars)"' <<< "${HF_TOKEN}"
fi

if [[ -n "${CF_TUNNEL_TOKEN:-}" ]]; then
    echo "==> Installing the Cloudflare tunnel token"
    "${SSH[@]}" "${TARGET}" \
        'umask 077; IFS= read -r t || true;
         printf "%s" "$t" > /workspace/.cloudflared-token;
         chmod 600 /workspace/.cloudflared-token;
         echo "   token installed (${#t} chars)"' <<< "${CF_TUNNEL_TOKEN}"
fi

if [[ -n "${MUSCRIPTOR_ALLOWED_ORIGINS:-}" ]]; then
    echo "==> Installing the browser-origin allowlist"
    "${SSH[@]}" "${TARGET}" \
        'umask 077; IFS= read -r origins || true; touch /workspace/.env;
         grep -v "^MUSCRIPTOR_ALLOWED_ORIGINS=" /workspace/.env \
             > /workspace/.env.new 2>/dev/null || true;
         printf "MUSCRIPTOR_ALLOWED_ORIGINS=%s\n" "$origins" \
             >> /workspace/.env.new;
         mv /workspace/.env.new /workspace/.env; chmod 600 /workspace/.env;
         echo "   allowlist installed"' <<< "${MUSCRIPTOR_ALLOWED_ORIGINS}"
fi

echo "==> Running setup.sh on the instance"
"${SSH[@]}" "${TARGET}" "bash ${REMOTE_DIR}/deploy/vastai/setup.sh"
