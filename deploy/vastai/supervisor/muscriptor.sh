#!/bin/bash
# Supervisor wrapper for the muscriptor transcription server.
#
# Vast.ai's base image runs long-lived processes under supervisor rather than
# systemd or Docker: the instance is an *unprivileged* container, so there is no
# Docker-in-Docker and the repo's own image (Dockerfile / swarm.yml) cannot be
# used here. The server therefore runs natively out of a uv venv, and this
# wrapper is what supervisor executes.
#
# The three `utils` includes are the image's conventions, not ours:
#   logging.sh      - routes our stdout into /var/log/portal/<name>.log
#   environment.sh  - exports /etc/environment plus ${WORKSPACE}/.env, which is
#                     where HF_TOKEN lives (see setup.sh). This is the only
#                     reason the token reaches the process, and it is why the
#                     token is never baked into this file or into git.
#   exit_portal.sh  - makes the service self-skip when its entry has been
#                     removed from /etc/portal.yaml, so the portal's
#                     enable/disable toggle works the way it does for the
#                     image's built-in services.

# Deliberately NO `set -euo pipefail` here, matching the image's own wrappers
# (/opt/supervisor-scripts/tensorboard.sh). `logging.sh` reads an optional log
# path from `$1`, so under `set -u` sourcing it dies with "$1: unbound variable"
# and supervisor reports a bare spawn error with no log to explain it.
utils=/opt/supervisor-scripts/utils
# shellcheck source=/dev/null
. "${utils}/logging.sh"
# shellcheck source=/dev/null
. "${utils}/environment.sh"
# shellcheck source=/dev/null
. "${utils}/exit_portal.sh" "MuScriptor"

APP_DIR="${MUSCRIPTOR_APP_DIR:-/workspace/muscriptor}"
# Bind to loopback only. Everything external arrives through the Caddy edge on
# the mapped public port, which is what applies token auth; binding 0.0.0.0
# would quietly publish an unauthenticated copy of the API on the same box.
BIND_HOST="${MUSCRIPTOR_BIND_HOST:-127.0.0.1}"
INTERNAL_PORT="${MUSCRIPTOR_INTERNAL_PORT:-18000}"
MODEL_SIZE="${MUSCRIPTOR_MODEL:-medium}"

cd "${APP_DIR}"

# `pty` is the image's helper (a shell function from utils/pty.sh, so it cannot
# be `exec`'d) that runs the server on a pseudo-terminal. That keeps Python's
# output line-buffered, so the model-load progress reaches the log live instead
# of sitting in a block buffer until the process exits.
pty ./.venv/bin/muscriptor serve \
    --host "${BIND_HOST}" \
    --port "${INTERNAL_PORT}" \
    --model "${MODEL_SIZE}" \
    --device cuda 2>&1
