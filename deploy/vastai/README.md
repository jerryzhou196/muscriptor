# Running the transcription server on a Vast.ai GPU instance

This directory provisions a rented [Vast.ai](https://vast.ai) GPU box to serve
muscriptor's **note transcription** API (`POST /transcribe`, SSE). It is an
*additional* deployment path — the Docker Swarm + Traefik setup in the repo root
(`Dockerfile`, `swarm.yml`, `deploy.sh`) is untouched and still the production
deployment.

Chord recognition is **not** here. It runs on CPU in a Hugging Face Docker Space,
because BTC-large-voca is cheap enough on CPU that it does not justify GPU rent.
This box only carries the expensive part.

The root `docker-compose.yml` is for a Docker-capable host. A standard Vast.ai
instance is itself an unprivileged container and cannot run Docker-in-Docker,
so this deployment uses the equivalent native service under Supervisor.

```
Vercel (static web UI)
   ├── /analyze     ──► HF Space  (CPU, free)      ← a different PR
   └── /transcribe  ──► this box  (GPU, ~$0.10-0.30/hr)
```

---

## Why there is no Terraform here

There is a usable community Terraform provider
(`realnedsanders/vastai` v0.3.1, published 2026-03-28 — `vastai_instance`,
`vastai_gpu_offers`, `vastai_volume`, …; there is no first-party one, and the
older `aalekhpatel07/vastai` is a dead 2024 stub). We deliberately do not use it.

The reason is that Terraform buys you very little against Vast.ai's actual
model. A Vast instance is a *spot rental on someone's desktop*: the host can
disappear, the IP and every port change on every re-rent, and the interesting
configuration lives **inside** the container rather than in the API object. What
you actually need is "get this box serving again", which is a provisioning
problem, not a state-reconciliation problem. So this is a plain SSH script.

What that costs: no plan/apply, no drift detection, no state file. Re-renting is
a manual step. See [Re-renting](#re-renting-after-destroy) below.

---

## Prerequisites

- A rented Vast.ai instance, running the **Vast PyTorch base image** (the setup
  script depends on that image's `supervisor`, `caddy`, and `vast-capabilities`
  tooling), with:
  - an **NVIDIA GPU with ≥ 8 GB VRAM** — the medium model plus CQT features fit
    comfortably in the 12 GB of an RTX 3060;
  - **at least 32 GB of disk**. 16 GB is too small and *will* fail partway
    through: see [Disk](#disk) below.
  - **at least two open TCP ports** beyond the defaults. External ports are
    allocated when the instance is created and **cannot be added later**.
- `rsync` and `ssh` locally.
- A Hugging Face token with the `MuScriptor/muscriptor-medium` licence accepted.
  The repo is `gated: auto` — public metadata, but an anonymous download of
  `model.safetensors` returns **401**.
- For a browser-facing deployment, a remotely managed Cloudflare Tunnel and a
  hostname in a Cloudflare-managed domain. The dashboard supplies the tunnel
  token; the published application route points at `http://localhost:18000`.

## Provision

```bash
# The host/port come from the Vast console's "Connect" line. The API origins
# are exact browser origins, not backend hostnames.
HF_TOKEN=hf_xxxxxxxxxxxx \
CF_TUNNEL_TOKEN=eyJ... \
MUSCRIPTOR_ALLOWED_ORIGINS=https://muscriptor.example.vercel.app \
  ./deploy/vastai/provision.sh root@203.0.113.10 17762
```

That rsyncs this checkout to `/workspace/muscriptor`, writes the Hugging Face
token and browser-origin allowlist to `/workspace/.env`, writes the tunnel
token to `/workspace/.cloudflared-token`, and runs [`setup.sh`](setup.sh). Both
token files are mode 600 and travel over SSH stdin, never argv. `setup.sh` is
also safe to run directly on the instance once the code is there.

In Cloudflare's tunnel dashboard, add a **Published application** route:

| Field | Value |
|---|---|
| Hostname | the fixed API name, for example `muscriptor-api.example.com` |
| Service URL | `http://localhost:18000` |
| Path | empty |

The origin URL is intentionally HTTP: it is loopback inside the container;
Cloudflare terminates browser-trusted HTTPS and carries the request through the
encrypted tunnel.

Follow the model load — it takes a minute or two on a cold cache:

```bash
ssh -p <port> root@<host> 'tail -f /var/log/portal/muscriptor.log'
```

## Health check

On the box (bypasses the auth edge, since it is loopback):

```bash
ssh -p <port> root@<host> 'curl -s localhost:18000/health'
# {"status":"ok"}
```

From outside, through Caddy — needs the instance token:

```bash
curl -H "Authorization: Bearer $OPEN_BUTTON_TOKEN" \
     http://<PUBLIC_IPADDR>:<VAST_TCP_PORT_10100>/health
```

`OPEN_BUTTON_TOKEN` is set inside the container and shown in the Vast console.

For the browser-facing hostname, no Vast token is sent:

```bash
curl https://muscriptor-api.example.com/health
# {"status":"ok"}
```

---

## How the frontend reaches this box

Vast's direct public port is HTTP, while Vercel serves the UI over HTTPS. A raw
Vast URL is therefore unusable in a browser: mixed-content rules block it. The
named Cloudflare Tunnel solves the transport side with a stable, trusted HTTPS
hostname, and `muscriptor serve` solves the browser side with its explicit
comma-separated `MUSCRIPTOR_ALLOWED_ORIGINS` allowlist.

Set Vercel's build-time variable to the published hostname:

```text
VITE_TRANSCRIBE_API_BASE=https://muscriptor-api.example.com
```

Set the matching frontend origin on the instance when provisioning:

```text
MUSCRIPTOR_ALLOWED_ORIGINS=https://muscriptor.example.vercel.app
```

The CORS middleware permits `GET`, `POST`, and the `X-Client-Id` header used by
`/transcribe`. An unlisted origin receives no CORS permission. Preview Vercel
deployments have generated origins and need to be allowlisted individually or
assigned a stable branch domain.

The tunnel routes directly to `127.0.0.1:18000`, bypassing Vast's Caddy bearer
authentication. **CORS is not authentication**: command-line clients can still
call a public hostname. Use Cloudflare rate limiting/WAF controls if exposing
the API broadly. Cloudflare Access login cannot simply be enabled because a
static browser bundle has nowhere safe to keep its service credential.

A Quick Tunnel is acceptable for a short demo, but its hostname changes when
the process restarts. The remotely managed named connector installed here runs
under Supervisor and keeps the dashboard-owned hostname stable across normal
process and instance restarts.

---

## What persists

There are **two** caches, and only one of them is under `/workspace`:

| Cache | Path | Holds |
|---|---|---|
| HuggingFace hub | `$HF_HOME` = `/workspace/.hf_home` | medium weights (~1.1 GB) + the 38 MB `.sf3` soundfont |
| muscriptor's plain-HTTP cache | `~/.cache/muscriptor` → `/root/.cache/muscriptor` | the BTC chord checkpoint |

The second one is easy to miss: `download_if_necessary` puts `hf://` URLs in
`HF_HOME` but plain `http(s)://` URLs — which is what the BTC checkpoint is — in
`~/.cache/muscriptor`. That is **outside the workspace**, so it is not covered
even when `/workspace` is a mounted volume. `setup.sh` prefetches it explicitly;
expect to re-download it on every fresh container.

Whether the HF cache survives depends on something you choose **when renting**:

```bash
vast-capabilities | jq '.instance.workspace_is_volume'
```

- `true` — `/workspace` is a host volume. The cache survives stop/start,
  **recycle, and destroy**. Rent with a volume if you re-provision often.
- `false` — **as on the instance this was developed against.** `/workspace` is
  ordinary container storage. It survives *stop/start* only; a **recycle or
  destroy wipes it** and the next run re-downloads 1.2 GB.

Nothing else on the box is precious: the venv and the code are both reproducible
from this repo.

### Disk

Budget **32 GB**, not 16. The venv is roughly 7 GB with the CUDA runtime and the
HF cache is another 1.2 GB; uv's download cache and temporary install space need
headroom beyond that. The setup script avoids installing parallel cu12 and cu13
runtime trees, but a 16 GB instance still leaves too little recovery margin for
future lockfile or model growth.

---

## The torch/CUDA correction

`setup.sh` derives a CUDA hold-back list from `uv.lock`, runs `uv sync` without
that stack, and then installs torch and torchaudio together from the compatible
PyTorch index. This is the single most important thing in the script.

`uv.lock` pins the current PyPI torch, which is a **CUDA 13.0** build. Vast hosts
very commonly run a 12.x driver — ours was 570.181, i.e. a CUDA 12.8 ceiling.
CUDA *major* forward-compatibility is a **datacenter-GPU-only** feature, so on
the consumer cards that make up most of the marketplace it does not apply. The
failure is quiet and expensive: torch imports fine, then reports

```
torch 2.13.0+cu130   cuda_available False
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
```

and the server serves **every request on CPU** — you are then paying GPU rent
for CPU inference, with nothing in the logs that looks like an error.

So the script reads the driver's ceiling from `nvidia-smi`, maps it to a
`download.pytorch.org/whl/cuXXX` index, and installs torch, torchaudio, and their
CUDA runtime from there. Application dependencies still come from the lock.
Verified on our box:

```
torch 2.11.0+cu128  cuda 12.8  available True  NVIDIA GeForce RTX 3060  cc (8,6)
```

Three things that do **not** work, all tried:

- `UV_TORCH_BACKEND=auto` with `uv sync` — the lockfile pins the exact wheel, so
  the backend hint is ignored and you get cu130 anyway.
- Adding `--extra-index-url https://pypi.org/simple` — uv resolves torch from the
  first index carrying the name (PyPI) and then reports `torch==2.11.0+cu128`
  as unsatisfiable.
- Syncing normally and *then* reinstalling torch — correct, but it downloads two
  ~2.5 GB CUDA builds and fills a 16 GB disk to 100%, dying with ENOSPC.

### torchaudio has to move with it

`torchaudio` is held back and installed from the same CUDA index in the same
command. TorchAudio 2.11's stable ABI supports PyTorch 2.11 and later, but the
runtime variant still has to be compatible; a wheel from a different CUDA
index can install cleanly and then fail at *import*:

```
OSError: Could not load this library: …/torchaudio/lib/_torchaudio.abi3.so
```

`beat-this` imports torchaudio, and beat detection runs inside `/transcribe`, so
the mismatch does not surface at startup. It surfaces as a request that streams
note events normally and then **dies before `transcription_complete`** — the
client sees a truncated SSE stream and the server logs an import traceback. If
you ever see that symptom, check that the two versions are in the supported
range and came from the same `cuXXX` index. `setup.sh` imports both plus
`beat_this` as a post-install check so this fails loudly at provision time
instead.

The hold-back list includes the locked `nvidia-*`, `cuda-*`, and `triton`
packages as well as torch and torchaudio. That prevents `uv sync` from first
installing the unused cu13 runtime and avoids a second multi-gigabyte CUDA tree
when the driver-matched wheels are installed.

---

## Idempotency

Re-running `provision.sh` / `setup.sh` is safe. Specifically:

| Step | On re-run |
|---|---|
| `rsync` | Converges (`--delete`). Edits on the box are overwritten. |
| `uv sync` (with the CUDA hold-back list) | Converges, and stays cheap: because the whole CUDA stack is held back, a re-run no longer tears out the cu12 packages and re-downloads the cu13 ones. |
| torch/torchaudio install | Converges — uv sees the correct builds already present and no-ops. |
| Weight prefetch | No-ops once cached. |
| Supervisor files | Overwritten in place (`install -m`). |
| `/workspace/.env` | Rewritten, not appended — old `MUSCRIPTOR_*` and `HF_TOKEN` lines are filtered out first, so no duplicates accumulate. |
| Tunnel token | Rewritten at `/workspace/.cloudflared-token` only when `CF_TUNNEL_TOKEN` is supplied. |
| `/etc/portal.yaml` | Converges, and **reuses the external port already assigned** to `MuScriptor` rather than consuming a new one. |
| `supervisorctl` | Restarts the service; brief downtime. |

Nothing accumulates and nothing needs a manual reset between runs. The one thing
to watch is disk: if a run ever dies with ENOSPC, clear the uv cache
(`uv cache clean`) before retrying rather than re-running blindly.

## Re-renting after destroy

`destroy` (and `recycle`) rebuilds the container from the image. Everything
outside a mounted volume is gone — the venv, `/etc/supervisor`,
`/etc/portal.yaml`, and `/workspace/.env` with it. There is no state file that
remembers any of this, which is the honest cost of not using Terraform.

To bring a new rental up:

1. Rent a new instance meeting the [prerequisites](#prerequisites) (GPU, **≥32 GB
   disk**, and spare open ports).
2. Re-run `provision.sh` with the **new** host and port, with `HF_TOKEN`, the
   existing `CF_TUNNEL_TOKEN`, and `MUSCRIPTOR_ALLOWED_ORIGINS` set.
3. Wait for the named connector to become healthy. The Cloudflare route and
   Vercel API base do not change because the tunnel identity and hostname stay
   fixed.
4. If you use the HF Space for chords, nothing there changes; the two services
   are independent.

## Cost

Rough, on-demand, at the time of writing:

- RTX 3060 / 3090-class, 1 GPU: **$0.10-0.30/hr** ≈ $70-220/month running
  continuously.
- A stopped instance still bills **storage** (a few $/month for 32 GB) but no GPU
  time. `vastai stop instance <id>` is the lever that matters.
- Interruptible (bid) instances are cheaper and can be preempted mid-request.
  Not recommended for a user-facing endpoint.

The model is small; a single mid-range GPU serves this comfortably. Paying for an
A100 here is wasted money.

## Operating

```bash
supervisorctl status muscriptor
supervisorctl status muscriptor_named_tunnel
supervisorctl restart muscriptor
tail -f /var/log/portal/muscriptor.log

# Halt GPU charges without losing the container filesystem:
vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY
```

## Verified against a live instance

This was developed against a real rented box, not written from documentation.
Measured there:

- **Host**: RTX 3060 12 GB, driver 570.181 (CUDA 12.8 ceiling), compute
  capability 8.6, 56 vCPU, 125 GB RAM, 16 GB disk, Vast PyTorch base image.
- **Stack**: torch 2.11.0+cu128 / torchaudio 2.11.0+cu128, `cuda_available True`.
- **Model load**: ~2.5 GB resident on the GPU.
- **`GET /health`** → `{"status":"ok"}` on loopback; `401` through the public
  port without a token, `{"status":"ok"}` with `Authorization: Bearer`.
- **`POST /transcribe`** on the repo's 15 s sample, `detect_tempo=best-effort`,
  `chords=true`: **6.9 s warm** (34 s on the first call, which downloads the BTC
  checkpoint), 173 note-start events, terminating correctly in a
  `transcription_complete` event carrying the MIDI with chord metadata.
- **Named tunnel + browser path**: four registered HTTP/2 tunnel connections;
  trusted HTTPS health and CORS preflight from the Vercel production origin;
  an unrelated origin received no CORS permission. A 10 s MP3 completed through
  the fixed hostname in 6.4 s with 351 SSE events and a non-empty MIDI payload.
- **Re-running `setup.sh`** converges, reuses the already-assigned external
  port, and leaves the service healthy.

## Teardown

```bash
vastai destroy instance <id> --api-key <key>   # or the Destroy button
```

There is nothing to clean up locally — no state file, no cloud resources beyond
the instance itself. Revoke the HF token if it was only for this box.
