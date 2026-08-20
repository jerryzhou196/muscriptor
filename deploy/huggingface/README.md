---
title: MuScriptor Chords
emoji: 🎸
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Chord recognition for MuScriptor, on CPU
---

# MuScriptor chord service

Chord recognition from audio, served as a small HTTP API from a free CPU Space.
Upload a recording to `POST /analyze` and get back the chord track — symbol,
start time, duration, and the notes each chord is made of — plus the beat grid
the chord boundaries were snapped to.

The model is [BTC](https://github.com/jayg996/BTC-ISMIR19) (Bi-directional
Transformer for Chord Recognition, Park et al., ISMIR 2019), large-vocabulary
checkpoint: 12 roots × 14 qualities plus "no chord". It runs on the CPU in a
couple of seconds per minute of audio, which is the whole reason this endpoint
exists separately — [MuScriptor](https://github.com/muscriptor/muscriptor)'s
note transcription needs a GPU, chord recognition does not, so the chord track
is served from here and the GPU is only rented when someone asks for notes.

Nothing musical is implemented in `app.py`. Chords, beat tracking and audio
decoding are imported from the `muscriptor` package installed into the image,
so this Space and the main server return the same chords for the same file.

## API

`https://<owner>-muscriptor-chords.hf.space`

### `GET /health`, `HEAD /health`

```json
{"status": "ok", "version": "0.3.0", "model": "BTC-large-voca"}
```

Cheap, never rate limited, and answers before the model has finished loading —
`"status"` is `"loading"` until it has. A free Space sleeps when it is idle, so
a client should ping `/health` to wake it before uploading anything, and wait
for `"ok"` before expecting `/analyze` to be quick.

### `POST /analyze` — `multipart/form-data`

| field | type | default | meaning |
|---|---|---|---|
| `file` | audio upload | required | any format libsndfile/ffmpeg reads (wav, mp3, flac, ogg, m4a, …) |
| `detect_tempo` | `"best-effort"｜"true"｜"false"` | `"best-effort"` | when a beat grid is found, chord boundaries snap to it. `"true"` fails the request when no grid is found, `"false"` skips beat tracking entirely |

```json
{
  "chords": [
    {"time": 0.0, "duration": 1.85, "label": "C", "root": 0, "intervals": [0, 4, 7]}
  ],
  "beat_grid": {"bpm": 120.0, "beats_per_bar": 4, "first_downbeat": 0.12, "onset_delay": null},
  "duration": 12.34,
  "processing_time_ms": 4210
}
```

- `time` and `duration` are seconds in the uploaded audio; `root` is a pitch
  class (0 = C) and `intervals` the semitones above it that sound, so a client
  can play the chord without knowing any music theory. Both are `null`/empty
  for an `"N.C."` span.
- The list starts at the first real chord. Silence a recording opens with is
  not a chord change; `"N.C."` spans *between* chords are kept, since they say
  the harmony stopped.
- Chord times need no correction — they are snapped to the beats already.
- `beat_grid` is `null` when no tempo was found. `onset_delay` is always `null`
  here: it measures how late transcribed *notes* sit against the beats, and
  this service transcribes none. The field is in the payload so that a client
  can consume one `beat_grid` shape from either backend.
- `label`, `root` and `intervals` are exactly what MuScriptor's own
  `/transcribe` puts in its `transcription_complete` event.

Errors are a sanitized `{"detail": "…"}`: `400` undecodable audio (or no beat
grid when `detect_tempo=true`), `401` missing/incorrect shared secret when one
is enforced, `413` too large or too long, `429` rate limited (with
`Retry-After`), `503` all analysis slots busy, `504` analysis timed out.
Details never quote a traceback, a package or a path — those go to the Space
logs instead.

## Configuration

Every knob is an environment variable, set under the Space's
**Settings → Variables and secrets**. Changing one restarts the Space; none of
them need a rebuild.

| variable | default | purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `https://muscriptor.vercel.app,http://localhost:5173` | CSV allowlist of browser origins for CORS. `*` is ignored on purpose: with it, any page on the internet could spend this Space's CPU from a visitor's browser. Set it to the real frontend origin for your deployment. |
| `RATE_LIMIT_WINDOW_SEC` | `60` | Length of the burst window, in seconds. |
| `RATE_LIMIT_PER_WINDOW` | `5` | Analyses one client may start per burst window. |
| `RATE_LIMIT_HOURLY` | `40` | Analyses one client may start per hour. |
| `MAX_CONCURRENT_ANALYSES` | `2` | Analyses running at once; requests beyond it get `503` rather than queueing behind a busy CPU. |
| `ANALYZE_TIMEOUT_SEC` | `180` | Wall-clock budget for one analysis, after which the client gets `504`. |
| `MAX_UPLOAD_MB` | `25` | Upload size cap, checked while reading so an oversized body is never fully buffered. |
| `MAX_AUDIO_SECONDS` | `600` | Duration cap, checked after decoding — a small file can still be an hour long. |
| `ENFORCE_EDGE_SHARED_SECRET` | `false` | When true, `/analyze` requires the shared secret header. |
| `EDGE_SHARED_SECRET` | *(unset)* | The secret itself. Store it as a **secret**, not a variable. |
| `EDGE_SHARED_SECRET_HEADER` | `X-Edge-Auth` | Header the secret is read from. |
| `ENABLE_PUBLIC_API_DOCS` | `false` | Swagger (`/docs`), ReDoc (`/redoc`) and `/openapi.json`. Off by default: they are a map of the API for anyone who finds the URL. |
| `TORCH_NUM_THREADS` | *(unset)* | Cap torch's thread pool. Worth setting to `1` when `MAX_CONCURRENT_ANALYSES` is above 1, since torch otherwise takes every core for each analysis. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

Rate limiting keys on the client IP as Hugging Face presents it: a Space never
sees the caller's socket, so the address is taken from the left-most entry of
`X-Forwarded-For` (`request.client.host` is the same HF router for everybody).
That header is client-controllable and an abuser can rotate it — the real
bounds on damage are `MAX_CONCURRENT_ANALYSES` and `ANALYZE_TIMEOUT_SEC`.

No Hugging Face token is needed at runtime. The BTC checkpoint comes from
GitHub (SHA-256 pinned) and the beat tracker's from a public HF repo; the gated
MuScriptor transcription weights are never loaded here.

## Deploying

The Space repository is generated from this one — do not edit it by hand, a
sync overwrites it.

1. Create the Space once: **New Space** on Hugging Face, SDK **Docker**,
   hardware **CPU basic (free)**, name it `muscriptor-chords` (any name works;
   it only has to match what the frontend calls).
2. Set `ALLOWED_ORIGINS` under Settings → Variables to your frontend's origin.
3. Add a Hugging Face **write** token
   ([settings/tokens](https://huggingface.co/settings/tokens)) as the `HF_TOKEN`
   secret of this GitHub repository.
4. Run the **Sync chord Space** workflow
   (`.github/workflows/sync-hf-space.yml`) from the Actions tab, giving it the
   Space as `owner/name`. It assembles the Space contents and force-pushes
   them; Hugging Face then builds the image.

The workflow lays the Space repository out as:

```
README.md          ← deploy/huggingface/README.md  (this file; its frontmatter configures the Space)
Dockerfile         ← deploy/huggingface/Dockerfile
.dockerignore      ← deploy/huggingface/.dockerignore
deploy/huggingface/app.py
pyproject.toml, uv.lock, LICENSE, LICENSE-BTC
muscriptor/        ← the package the service imports
```

Hugging Face insists on `README.md` and `Dockerfile` at the repository root;
everything else keeps the layout it has in this repository. **The build context
is the repository root**, which is why the Dockerfile's `COPY` paths read
`pyproject.toml`, `muscriptor/` and `deploy/huggingface/app.py` — the same
paths work from a checkout here:

```bash
docker build -f deploy/huggingface/Dockerfile -t muscriptor-chords .
docker run --rm -p 7860:7860 muscriptor-chords
curl -F file=@web/public/headache_by_lost_deposit_10s.mp3 http://localhost:7860/analyze
```

To push without GitHub Actions, do by hand what the workflow does: copy those
files into a clone of the Space and `git push`.

### About the image

- `python:3.11-slim`, ffmpeg + libsndfile1, running as uid 10001. Every cache
  the process writes (`HOME`, `HF_HOME`, numba's) is redirected under
  `/home/app`, since a Space's non-root user cannot write `/root/.cache`.
- torch and torchaudio are installed from PyTorch's **CPU** index, pinned by
  the `TORCH_VERSION` / `TORCHAUDIO_VERSION` build args to what `uv.lock`
  resolves. The PyPI wheels bundle over 2 GB of CUDA that this Space would
  never use. Bump the two together when the lockfile moves.
- The rest of the dependencies are resolved from `pyproject.toml` rather than
  installed from `uv.lock`: the lock pins the PyPI (CUDA) build of torch, which
  would undo the line above. The trade-off is that a Space image is not
  bit-identical to a `uv sync` install — everything but torch floats within the
  ranges `pyproject.toml` declares.
- The BTC checkpoint is downloaded and digest-checked **during the build**, the
  way the repository's root Dockerfile pre-warms its soundfonts, so a cold
  container answers its first request without fetching weights. The beat
  tracker's checkpoint is pre-fetched too, best-effort.
- There is no `chords` extra in `pyproject.toml`, and adding one would not make
  this image smaller: everything MuScriptor declares as a Python dependency is
  also needed here (torch, librosa, soundfile, beat-this, mido, fastapi …).
  What this Space does *not* need — MuseScore, fluidsynth, the soundfonts, the
  web build — are system packages and build steps, not Python dependencies, and
  they are simply absent from this Dockerfile.

## If the Space is being abused

Symptoms: the queue is permanently full, `429`s in the logs from a handful of
addresses, or the free CPU quota disappearing. In rough order of how much they
cost a legitimate user, all of them Settings → Variables, no rebuild:

1. `TORCH_NUM_THREADS=1` and `MAX_CONCURRENT_ANALYSES=1` — the Space stays up
   and serves one analysis at a time.
2. `RATE_LIMIT_PER_WINDOW=2`, `RATE_LIMIT_HOURLY=10` — a normal session still
   works; a script does not.
3. `MAX_AUDIO_SECONDS=120`, `MAX_UPLOAD_MB=10`, `ANALYZE_TIMEOUT_SEC=60` — caps
   the cost of any single request.
4. `ALLOWED_ORIGINS` down to the production origin alone (drop
   `http://localhost:5173`). This only stops browsers, not `curl`.
5. `ENFORCE_EDGE_SHARED_SECRET=true` with `EDGE_SHARED_SECRET` set (and the
   same value configured in the frontend) — closes the endpoint to everything
   but the deployment. This is the switch that actually stops a determined
   caller, since IP-based limits are only as good as `X-Forwarded-For`.
6. Set the Space to **private** in its settings, or pause it, and let the
   frontend fall back to transcribing without a chord track.
