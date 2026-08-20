# Infrastructure

How a hosted muscriptor is put together: what runs where, why it is split that
way, and what an operator has to set to stand it up.

The repository still ships the single-box deployment it always had — one
container serving the API and the built frontend from the same origin, behind
Traefik on a Docker Swarm host (`Dockerfile`, `swarm.yml`, `deploy.sh`,
`.github/workflows/deploy.yml`). Nothing here replaces it. This document
describes the *split* deployment, which exists because the two models in this
project have very different appetites.

## The shape of it

```
                         ┌──────────────────────────────┐
   browser ──────────────▶  Vercel — static web UI      │
                         │  muscriptor/web, Vite bundle │
                         └───────┬──────────────┬───────┘
                                 │              │
              POST /analyze      │              │   HTTPS via named tunnel
              GET  /health       │              │   GET  /instruments
                                 │              │   POST /transcribe (SSE),
                                 │              │   /auralize, /sheets
                                 ▼              ▼
        ┌────────────────────────────┐   ┌─────────────────────────────────┐
        │ Hugging Face Docker Space  │   │ Vast.ai GPU instance            │
        │ CPU basic (free tier)      │   │ set up over SSH                 │
        │                            │   │                                 │
        │ BTC-large-voca chord model │   │ muscriptor transcription model  │
        │ torch on CPU               │   │ torch on CUDA                   │
        └────────────────────────────┘   └─────────────────────────────────┘
```

Three tiers, because each part wants different hardware:

- **The web UI is static.** Once the API base URLs are baked in at build time
  it is a bundle of hashed assets, which is exactly what a CDN is for. Vercel
  gives it a global edge, per-branch preview deployments, and TLS without an
  operator touching a certificate.
- **Chord recognition does not need a GPU.** BTC reads a constant-Q
  spectrogram and labels ~93 ms frames; it is small, and CPU inference is
  comfortably fast enough for a few-minute song. Putting it on a free-tier
  Hugging Face Space means the chord feature costs nothing to run and stays up
  whether or not the GPU box is alive. This mirrors the reference deployment
  this design was copied from, which serves the same BTC-large-voca checkpoint
  from a CPU Space.
- **Note transcription does need a GPU**, and rented GPU-hours are the whole
  cost of the system. Vast.ai is a spot market for them: an instance is rented
  by constraint rather than as a fixed instance type, set up over SSH, and
  expected to be torn down when idle.

The corollary is that the tiers fail independently, and the UI is written to
expect it. A sleeping or missing chord service must not stop a transcription —
chords are opt-in in the UI already. A missing GPU tier is fatal to
transcription, and that is the failure the operator gets paged for.

## Tier 1 — Vercel, the web UI

Built from `muscriptor/web` with Vite. Two build-time variables select the
backends; both default to empty, which means *same origin* and preserves the
existing single-box deployment byte-for-byte.

| Variable | Points at | Unset means |
|---|---|---|
| `VITE_TRANSCRIBE_API_BASE` | the Vast.ai instance | same-origin `/transcribe`, … |
| `VITE_CHORD_API_BASE` | the HF Space | chords come from the SSE event, as before |

Set them in the Vercel project's environment variables. Because they are
inlined into the bundle at build time, changing one requires a redeploy, not
just a restart.

In the deployment described here **both variables are set**. Hugging Face
provides TLS for the chord service. A Cloudflare named tunnel provides the GPU
box with a stable, browser-trusted HTTPS hostname, while the MuScriptor
server's `MUSCRIPTOR_ALLOWED_ORIGINS` setting allows the Vercel origin and its
`X-Client-Id` request header.

**Preview deployments are the operational trap.** Every branch gets its own
`*.vercel.app` origin, and both backends run strict CORS allowlists rather than
`*`. A preview deployment therefore cannot call the production backends until
its origin is added to `ALLOWED_ORIGINS` and `MUSCRIPTOR_ALLOWED_ORIGINS`.
Prefer a stable branch domain or temporarily allowlist the exact preview URL.

## Tier 2 — Hugging Face, the chord service

A Docker Space (`sdk: docker`, `app_port: 7860`) built from
`deploy/huggingface/`, wrapping `muscriptor.utils.chords` in a small FastAPI
app. It exposes exactly two routes.

**It needs no Hugging Face credentials to run.** The BTC checkpoint is fetched
from GitHub and checked against `CHECKPOINT_SHA256` before it is unpickled, so
nothing here touches a gated Hub repo — unlike the transcription weights and
the soundfonts, which are `hf://` URLs and do need a token. A write token is
required only to *create and push* the Space, and by
`.github/workflows/sync-hf-space.yml` to do that from CI.

### `GET /health`, `HEAD /health`

```json
{"status": "ok", "version": "0.3.0", "model": "BTC-large-voca"}
```

Cheap, never rate limited, and answers `{"status": "loading"}` while the model
is still coming up. **Free Spaces sleep after inactivity and take tens of
seconds to wake**, so the UI pings this as early as it can and treats a slow or
failed answer as "chords unavailable", not as an error worth showing.

### `POST /analyze` — `multipart/form-data`

| field | type | default |
|---|---|---|
| `file` | audio upload, any format libsndfile/ffmpeg reads | required |
| `detect_tempo` | `"best-effort"` \| `"true"` \| `"false"` | `"best-effort"` |

```json
{
  "chords": [
    {"time": 0.12, "duration": 0.8, "label": "F#", "root": 6, "intervals": [0, 4, 7]}
  ],
  "beat_grid": {"bpm": 154.857, "beats_per_bar": 4, "first_downbeat": 0.12, "onset_delay": null},
  "duration": 10.02,
  "processing_time_ms": 2609
}
```

`beat_grid` is `null` when no tempo was found. `label`, `root` and `intervals`
are the same fields the server already puts in its `transcription_complete` SSE
event, deliberately — the frontend keeps one chord type no matter which tier
produced it. Chord times need no `onset_delay` correction; they were snapped to
the beats, not to the notes.

`onset_delay` is always `null` from this tier, and that is not an omission:
it measures how late transcribed *notes* sit against the beats, and this
service transcribes no notes. The field stays in the payload so that the UI has
one `beat_grid` shape to consume regardless of which tier answered.

Errors come back as FastAPI `{"detail": "…"}` with the detail sanitized:
`400` undecodable audio, `413` too large or too long, `429` rate limited, `503`
all analysis slots busy, `504` analysis timed out.

### Why it is hardened

The Space is public on purpose, and has to be. A private Space serves only
requests carrying an `Authorization: Bearer` token, and the frontend is a
static bundle with nowhere to keep one — Vite inlines `VITE_*` variables into
the JavaScript it ships, so anything put there is readable by anyone who opens
the page. That rules out authenticating the browser to the Space at all, and it
is why the defences below are about *rate* and *origin* rather than identity.
The same limit applies to `EDGE_SHARED_SECRET`: it is only meaningful if a
server-side proxy holds it, not the browser.

So: a public, unauthenticated endpoint on hardware someone else pays for,
configured defensively out of the box. See
`deploy/huggingface/README.md` for the full list and defaults; the shape is a
CORS allowlist (never `*`), a per-client burst-plus-hourly rate limit, a cap on
concurrent analyses, a hard analysis timeout, upload size and duration caps,
API docs off, and an optional shared-secret header if the endpoint should only
be reachable through the frontend.

## Tier 3 — Vast.ai, the GPU instance

A GPU instance rented on Vast.ai by constraint (GPU model, VRAM, disk, region,
maximum $/hr) and set up over SSH. A Vast instance is already an unprivileged
container, so the setup builds a native uv environment and runs `muscriptor
serve` under the base image's Supervisor. See `deploy/vastai/README.md` for the
setup, tunnel, and teardown cycle.

This tier is provisioned by hand rather than declared. That is a deliberate
trade for a single rented spot instance: Vast.ai has no first-class Terraform
provider, an instance's identity does not survive being destroyed and
re-rented, and a scripted SSH setup is honest about being a one-shot rather
than pretending to reconcile state it cannot observe. The cost is that nothing
detects drift, and re-renting means re-running the setup.

### The browser reaches it through a named tunnel

Vast.ai's direct `IP:port` endpoint is plain HTTP and cannot be called from an
HTTPS Vercel page. The deployment runs a remotely managed Cloudflare Tunnel
connector under Supervisor and publishes a fixed hostname whose dashboard
route points to `http://localhost:18000`. Cloudflare terminates public HTTPS;
the local HTTP hop stays on loopback and travels through the encrypted tunnel.

The browser calls that hostname directly through `VITE_TRANSCRIBE_API_BASE`.
`muscriptor/server.py` installs CORS middleware only when
`MUSCRIPTOR_ALLOWED_ORIGINS` contains exact frontend origins. It allows `GET`,
`POST`, and `X-Client-Id`; a different origin receives no CORS permission.

The named tunnel bypasses Vast's token-authenticated Caddy edge. CORS is a
browser policy, not API authentication, so Cloudflare rate limiting or WAF
rules are the appropriate protection for a public endpoint. Cloudflare Access
cannot be added without a separate browser-auth design: a static Vite bundle
cannot safely hold a tunnel service credential.

### Whatever the route, two things matter

- **`HF_TOKEN` is required** — the transcription weights are private. It is
  passed to the container and never committed.
- **The HF hub cache must be on a persistent volume.** The weights and the
  253 MB of soundfonts are re-downloaded on every cold container otherwise,
  which is slow and, on a metered instance, wasteful.

## Secrets and where they live

| Secret | Held by | Used for |
|---|---|---|
| `HF_TOKEN` | the GPU instance's environment; a GitHub Actions secret | pulling the gated transcription weights; pushing the Space from CI |
| Cloudflare tunnel token | `/workspace/.cloudflared-token` on the GPU instance | attaching the Supervisor connector to the fixed API hostname |
| `EDGE_SHARED_SECRET` | HF Space secret, plus whatever proxy holds the other half | optional, locks `/analyze` to a server-side caller |

Note that `HF_TOKEN` wears two hats: a *read* token is what the GPU tier needs
for the gated weights, while pushing the Space needs a *write* token. They do
not have to be the same token, and the one in CI should be write-scoped and
nothing more.

None of these belong in the repository.

## Bringing it up

1. **GPU tier first** — create the remotely managed Cloudflare Tunnel and its
   published application route, then rent and provision the Vast.ai instance
   with its connector token and the Vercel origin. Wait for both Supervisor
   services and `curl https://<gpu-hostname>/health` to become healthy.
2. **Chord tier** — create the Space, push `deploy/huggingface/` (by hand or
   via the sync workflow), wait for the build, then `curl <space>/health` until
   `status` is `ok` rather than `loading`.
3. **Web tier last** — set `VITE_TRANSCRIBE_API_BASE` to the fixed tunnel
   hostname and
   `VITE_CHORD_API_BASE` in Vercel and deploy. They are inlined at build time,
   so this is the step that has to come after the other two have URLs.
4. **Close both CORS loops** — add the Vercel production origin to the Space's
   `ALLOWED_ORIGINS` and the GPU instance's `MUSCRIPTOR_ALLOWED_ORIGINS`. Until
   this is done each service can look healthy from `curl` while browser calls
   are blocked.

Tearing down is the reverse, and destroying the Vast.ai instance is the step
that actually stops the spending — the other two tiers are free.
