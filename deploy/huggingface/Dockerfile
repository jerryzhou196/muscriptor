# Image for the "muscriptor-chords" Hugging Face Docker Space (see the README
# next to this file). The build context is the *repository root*, not this
# directory: the image installs the real `muscriptor` package from the checkout
# so the Space runs the same chord recognition code as the main server rather
# than a copy of it. Build it locally with:
#
#   docker build -f deploy/huggingface/Dockerfile -t muscriptor-chords .
#
# and the sync workflow lays the Space repo out the same way (this file copied
# to the Space root, the rest of the tree kept where it is).

FROM python:3.11-slim

# ffmpeg and libsndfile1 are what soundfile/librosa call out to for anything
# that is not a PCM WAV — mp3, m4a, ogg, flac. Everything else the service
# needs is a wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# A Space runs as a non-root user, so every path the process writes to has to
# belong to it: HOME (where `muscriptor.utils.download` caches the BTC
# checkpoint), the HF hub cache (where beat_this's checkpoint lands — the
# default /root/.cache is not writable here) and numba's cache, which librosa
# fills on first use.
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    XDG_CACHE_HOME=/home/app/.cache \
    NUMBA_CACHE_DIR=/home/app/.cache/numba \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --uid 10001 --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /home/app/.cache /app \
    && chown -R 10001:10001 /home/app /app

WORKDIR /app

# torch and torchaudio come from PyTorch's CPU index, pinned to the versions
# uv.lock resolves. The default PyPI wheels bundle CUDA — over 2 GB of it — for
# a Space that will only ever run `torch.device("cpu")`. Bump these together
# when the lockfile moves; they must stay a matched pair.
ARG TORCH_VERSION=2.13.0
ARG TORCHAUDIO_VERSION=2.11.0
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

# uv is the project's package manager; here it is used only as a fast pip that
# can install a pyproject's dependencies without its source, so the dependency
# layer survives every change to muscriptor's own code. Nothing resolves torch
# again: the CPU build above already satisfies the `torch>=2.3` floor.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

# README.md and LICENSE are not documentation here — hatchling reads both while
# building the wheel (`readme` and `license-files` in pyproject.toml), and the
# build fails without them.
COPY README.md LICENSE ./
COPY muscriptor/ ./muscriptor/
COPY deploy/huggingface/app.py ./app.py
RUN uv pip install --system --no-cache --no-deps .

USER 10001

# Bake the model weights into the image, the way the root Dockerfile pre-warms
# the soundfonts: a Space that has gone to sleep must not spend its first
# request downloading 12 MB of checkpoint and verifying it. `_checkpoint_path`
# is the same download-and-check-the-digest path the service uses at runtime,
# so a tampered or truncated download fails the build rather than the request.
# The beat tracker's checkpoint is best-effort — it is a separate host, and
# chords without a beat grid are still chords.
RUN python -c "from muscriptor.utils.chords import _checkpoint_path; print('BTC checkpoint at', _checkpoint_path())" \
    && (python -c "from beat_this.inference import Audio2Beats; Audio2Beats(checkpoint_path='final0', device='cpu')" \
        || echo 'warning: beat tracker checkpoint not pre-fetched; it downloads on first use')

# 7860 is the port Hugging Face routes a Docker Space to (app_port in the
# README frontmatter). One worker on purpose: the concurrency cap, the rate
# limiter and the loaded model are all per-process state.
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
