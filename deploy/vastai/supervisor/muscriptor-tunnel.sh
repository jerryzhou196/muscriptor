#!/bin/bash
# Connect the loopback-only MuScriptor API to a remotely managed Cloudflare
# Tunnel. The dashboard owns the hostname -> http://localhost:18000 route;
# this process only authenticates the connector and keeps it online.

utils=/opt/supervisor-scripts/utils
# shellcheck source=/dev/null
. "${utils}/logging.sh"
# shellcheck source=/dev/null
. "${utils}/environment.sh"

TOKEN_FILE="${MUSCRIPTOR_TUNNEL_TOKEN_FILE:-/workspace/.cloudflared-token}"
if [[ ! -s "${TOKEN_FILE}" ]]; then
    echo "Cloudflare tunnel token file is missing or empty: ${TOKEN_FILE}" >&2
    exit 1
fi

exec /opt/instance-tools/bin/cloudflared tunnel \
    --no-autoupdate \
    --protocol http2 \
    run --token-file "${TOKEN_FILE}" 2>&1
