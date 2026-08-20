"""Chord recognition as a standalone HTTP service, for a Hugging Face Space.

BTC runs comfortably on a CPU, which the note transcription model does not, so
the chord track is split off the main server and served from a free HF Docker
Space: the static web UI calls this Space for chords and only rents a GPU when
someone asks for notes.

Nothing musical is implemented here. The chords, the beat grid and the audio
decoding all come from the installed `muscriptor` package, and the JSON is
shaped exactly like the `chords` / `beat_grid` fields of the
`transcription_complete` SSE event `muscriptor/server.py` already emits, so the
web UI keeps a single chord type no matter which backend answered.

What *is* implemented here is everything a public, unauthenticated, free-tier
endpoint needs and an in-house server behind a VPN does not: an origin
allowlist, per-client rate limits, a cap on how many analyses run at once, a
wall-clock timeout, upload size and duration caps, an optional shared secret,
and error bodies that say what the client did wrong without saying anything
about this machine. Every one of those is an environment variable, so the knobs
can be turned down from the Space's settings page during an incident without a
rebuild — see the README.
"""

import asyncio
import hmac
import io
import logging
import os
import time
import wave
from collections import defaultdict, deque
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Annotated

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from muscriptor.utils.audio import _read_non_wav_file, _read_wav_file, resample
from muscriptor.utils.beats import BeatDetectionError, TempoDetection, detect_grid
from muscriptor.utils.chords import detect_chords, prefers_flats, published_chords

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("muscriptor.chords")

# The rate the audio is handed to both models at. `TranscriptionModel._load_wav`
# resamples to this before calling either of them, and the beat tracker and BTC
# both resample again internally from whatever they are given — so feeding them
# 16 kHz mono here is what makes this service's output identical to the main
# server's for the same file, rather than merely similar.
SAMPLE_RATE = 16000

MODEL_NAME = "BTC-large-voca"

try:
    VERSION = package_version("muscriptor")
except PackageNotFoundError:  # running from a source checkout without an install
    VERSION = "0.0.0+unknown"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        logger.warning("%s is not an integer; using %d", name, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return _env_str(name, "true" if default else "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Browsers are the only intended caller, so the allowlist is the deployment's
# actual origins and never "*": with "*" any page on the internet could spend
# this Space's CPU budget from a visitor's browser. The default covers the
# production frontend and a local `vite dev`; a Space deployed under a
# different domain must set ALLOWED_ORIGINS.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in _env_str(
        "ALLOWED_ORIGINS", "https://muscriptor.vercel.app,http://localhost:5173"
    ).split(",")
    if origin.strip() and origin.strip() != "*"
]

# Two limits per client, because they defend against different things: the
# window catches a script hammering the endpoint, the hourly budget catches a
# patient one that stays just under it. Both are deliberately small — a single
# analysis is seconds of CPU on a two-core free-tier box.
RATE_LIMIT_WINDOW_SEC = _env_int("RATE_LIMIT_WINDOW_SEC", 60)
RATE_LIMIT_PER_WINDOW = _env_int("RATE_LIMIT_PER_WINDOW", 5)
RATE_LIMIT_HOURLY = _env_int("RATE_LIMIT_HOURLY", 40)

# How many analyses may hold the CPU at once, and how long any one of them may
# take. The Space has two cores; letting a third request in only makes all
# three slower and pushes them all towards the timeout.
MAX_CONCURRENT_ANALYSES = _env_int("MAX_CONCURRENT_ANALYSES", 2)
ANALYZE_TIMEOUT_SEC = _env_int("ANALYZE_TIMEOUT_SEC", 180)

# Two caps rather than one: bytes are checked before anything is decoded (a
# 300 MB upload must not be buffered to find out it is too big), seconds are
# checked after, since a small FLAC can still be an hour long.
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 25)
MAX_AUDIO_SECONDS = _env_int("MAX_AUDIO_SECONDS", 600)

# An optional shared secret the Vercel frontend (or an edge function in front
# of it) sends along, so the Space can be closed to everything but the
# deployment without standing up real auth. Off by default: the Space is
# useless without it only if EDGE_SHARED_SECRET is also set.
ENFORCE_EDGE_SHARED_SECRET = _env_bool("ENFORCE_EDGE_SHARED_SECRET", False)
EDGE_SHARED_SECRET = os.environ.get("EDGE_SHARED_SECRET", "")
EDGE_SHARED_SECRET_HEADER = _env_str("EDGE_SHARED_SECRET_HEADER", "X-Edge-Auth")

# Interactive docs are a map of the API for anyone who finds the Space URL;
# they are off unless someone explicitly wants them while developing.
ENABLE_PUBLIC_API_DOCS = _env_bool("ENABLE_PUBLIC_API_DOCS", False)

# torch defaults to one thread per core, which oversubscribes the box as soon
# as two analyses run at once. Unset means "leave torch alone".
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 0)
if TORCH_NUM_THREADS > 0:
    torch.set_num_threads(TORCH_NUM_THREADS)

# CPU only, always: this Space is on the free tier and BTC does not need more.
DEVICE = torch.device("cpu")


class _RateLimiter:
    """Per-client request timestamps, checked against a burst and an hourly cap.

    In-memory and per-process, which is all a single-container Space can do —
    and enough, since the thing being protected is this container's own CPU.
    Clients that stop calling are forgotten on the next sweep so a stream of
    one-shot IPs cannot grow the table without bound.
    """

    def __init__(self) -> None:
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._last_sweep = 0.0

    async def check(self, client: str) -> None:
        """Record a request from `client`, raising 429 if it is over a limit."""
        now = time.monotonic()
        async with self._lock:
            self._sweep(now)
            hits = self._hits[client]
            while hits and now - hits[0] > 3600:
                hits.popleft()
            in_window = sum(1 for hit in hits if now - hit <= RATE_LIMIT_WINDOW_SEC)
            if in_window >= RATE_LIMIT_PER_WINDOW:
                retry_after = RATE_LIMIT_WINDOW_SEC
            elif len(hits) >= RATE_LIMIT_HOURLY:
                retry_after = int(3600 - (now - hits[0])) + 1
            else:
                hits.append(now)
                return
        logger.warning(
            "rate limited %s (%d in window, %d in hour)", client, in_window, len(hits)
        )
        raise HTTPException(
            status_code=429,
            detail="too many requests, try again later",
            headers={"Retry-After": str(max(1, retry_after))},
        )

    def _sweep(self, now: float) -> None:
        """Drop clients with nothing left inside the hourly window."""
        if now - self._last_sweep < 300:
            return
        self._last_sweep = now
        for client in [
            c for c, hits in self._hits.items() if not hits or now - hits[-1] > 3600
        ]:
            del self._hits[client]


_rate_limiter = _RateLimiter()
_analysis_slots = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

# Flipped by the background warm-up below. /health reports "loading" until then
# rather than blocking, because the UI pings /health to wake a sleeping Space
# and needs an answer within its own timeout, not after the model has loaded.
_model_ready = False


def _load_models() -> None:
    """Load BTC (and pull the beat tracker's imports in) once, off the loop.

    Both are cached module-side by the muscriptor package — `chords.load_model`
    keeps the network per device, and beat_this caches its checkpoint on disk —
    so this is purely about paying the cost at startup instead of inside the
    first request. The checkpoint itself is baked into the image (see the
    Dockerfile), so this normally touches nothing but the local cache.
    """
    from muscriptor.utils.chords import load_model

    load_model(DEVICE)
    try:
        # Importing beat_this is what drags in torchaudio and soxr — seconds of
        # import time that would otherwise land on whoever asks for a beat grid
        # first. Not fatal: chords without a grid are still chords.
        import beat_this.inference  # noqa: F401
    except Exception:
        logger.exception("beat tracker unavailable; /analyze will return no beat grid")


async def _warm_up() -> None:
    global _model_ready
    started = time.monotonic()
    try:
        await asyncio.to_thread(_load_models)
    except Exception:
        # Left not-ready on purpose: /health keeps reporting "loading", which is
        # what the UI shows as "still waking up" rather than a broken analysis.
        logger.exception("model warm-up failed")
        return
    _model_ready = True
    logger.info("models ready in %.1fs", time.monotonic() - started)


async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_warm_up())
    yield
    task.cancel()


app = FastAPI(
    title="muscriptor chord service",
    version=VERSION,
    lifespan=_lifespan,
    # Both the UI and the schema it is generated from, so a probe of /docs,
    # /redoc or /openapi.json finds nothing to enumerate.
    docs_url="/docs" if ENABLE_PUBLIC_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_PUBLIC_API_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_PUBLIC_API_DOCS else None,
)


@app.middleware("http")
async def _reject_oversized_bodies(request: Request, call_next):
    """Refuse an over-large upload before anything parses it.

    This has to be middleware rather than a check in the endpoint: by the time
    a handler with an `UploadFile` parameter runs, Starlette has already read
    the whole multipart body (spooling it to a temp file). Rejecting on the
    declared Content-Length is what keeps a 300 MB upload from being buffered
    just to be told it was too big; the byte counting in `_read_capped` then
    catches bodies that declared nothing or lied.

    The allowance over the cap is multipart framing — boundaries, headers, the
    other form fields — which is a few hundred bytes, not megabytes.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit():
        if int(declared) > MAX_UPLOAD_MB * 1024 * 1024 + 64 * 1024:
            return JSONResponse(
                status_code=413,
                content={"detail": f"audio file is larger than {MAX_UPLOAD_MB} MB"},
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # nothing here is per-user; no cookies, no tokens
    allow_methods=["GET", "HEAD", "POST", "OPTIONS"],
    allow_headers=["Content-Type", EDGE_SHARED_SECRET_HEADER],
    max_age=600,
)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence against leaking this machine into a response body.

    FastAPI's default 500 page is fine, but anything that reaches it has an
    exception message attached in the logs only if we put it there — so log the
    detail (with the traceback) and hand the client a sentence that says
    nothing about paths, packages or filenames.
    """
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


def _client_id(request: Request) -> str:
    """Who to rate limit, as best as this can be known behind the HF router.

    A Space never sees the caller's socket: requests arrive from Hugging
    Face's front end, so `request.client.host` is the same proxy address for
    everyone and rate limiting on it would throttle the whole world together.
    The origin address is the first entry of X-Forwarded-For (the proxy chain
    appends its own peer to the right, so the left-most entry is the client).
    It is client-controllable — a determined abuser can rotate it — but the
    only alternative is no per-client limit at all; MAX_CONCURRENT_ANALYSES
    and the timeout are what actually bound the damage.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


def _check_shared_secret(request: Request) -> None:
    """Enforce the edge secret, when the deployment has turned it on."""
    if not ENFORCE_EDGE_SHARED_SECRET:
        return
    if not EDGE_SHARED_SECRET:
        # Misconfiguration, not a client error: refusing everything is the safe
        # reading of "enforce a secret I was never given".
        logger.error(
            "ENFORCE_EDGE_SHARED_SECRET is set but EDGE_SHARED_SECRET is empty"
        )
        raise HTTPException(status_code=503, detail="service unavailable")
    presented = request.headers.get(EDGE_SHARED_SECRET_HEADER.lower(), "")
    if not hmac.compare_digest(presented, EDGE_SHARED_SECRET):
        raise HTTPException(status_code=401, detail="unauthorized")


async def _read_capped(file: UploadFile) -> bytes:
    """The upload's bytes, refusing anything over MAX_UPLOAD_MB with a 413.

    The declared size was already checked by `_reject_oversized_bodies`; this
    is the same cap applied to what actually arrived, which is what catches a
    body that declared no length or under-declared it.
    """
    limit = MAX_UPLOAD_MB * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413, detail=f"audio file is larger than {MAX_UPLOAD_MB} MB"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode(data: bytes, filename: str | None) -> tuple[torch.Tensor, int]:
    """Decode an upload to (wav [C, T], sample rate), exactly as /transcribe does.

    PCM WAV goes through the stdlib reader (keeping WAV decoding byte-for-byte
    identical to the CLI); anything that isn't a readable WAV — mp3, flac, ogg,
    m4a, … — falls back to soundfile/libsndfile. A genuinely undecodable upload
    is the client's fault, so it is a 400 rather than a 500; unlike the main
    server, the decoder's own message stays in the log, since libsndfile's
    errors quote paths and internals.
    """
    try:
        return _read_wav_file(io.BytesIO(data))
    except (wave.Error, EOFError):
        try:
            return _read_non_wav_file(io.BytesIO(data))
        except Exception as e:
            logger.info("undecodable upload %r: %s", filename, e)
            raise HTTPException(
                status_code=400, detail="could not decode the uploaded audio file"
            ) from e


def _to_mono_16k(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """[C, T] at any rate to [1, T] at SAMPLE_RATE — `_load_wav`'s tensor path."""
    wav = wav.float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        wav = resample(wav, sample_rate, SAMPLE_RATE)
    return wav


def _analyze_audio(wav: torch.Tensor, detect_tempo: TempoDetection) -> dict:
    """Beat grid then chords for one mono 16 kHz waveform. Runs in a worker thread.

    The order matters: the grid is detected first so the chord boundaries can be
    snapped to it, which is what puts a chord change on a bar line instead of a
    32nd note off one. Without a grid the boundaries stay where the model heard
    them, and `beat_grid` comes back null.
    """
    grid = None
    if detect_tempo is not False:
        try:
            grid = detect_grid(wav, SAMPLE_RATE)
        except BeatDetectionError as e:
            if detect_tempo is True:
                # The client asked for a grid or nothing; this is about the
                # audio it sent, so it is safe (and useful) to say so.
                raise HTTPException(status_code=400, detail=f"no beat grid: {e}") from e
            logger.info("no beat grid (%s); continuing without one", e)

    recognized = detect_chords(wav, SAMPLE_RATE, grid, device=DEVICE)
    # Decided once over every span, so the symbols are spelled the way the main
    # server would spell them for the same recording.
    flats = prefers_flats(recognized)
    return {
        "chords": [
            {
                "time": chord.start,
                "duration": chord.end - chord.start,
                "label": chord.label(flats),
                "root": chord.root,
                "intervals": list(chord.intervals),
            }
            for chord in published_chords(recognized)
        ],
        "beat_grid": {
            "bpm": grid.bpm,
            "beats_per_bar": grid.beats_per_bar,
            "first_downbeat": grid.first_downbeat,
            # Always null here, and that is not an omission: onset_delay is how
            # late the *transcribed notes* sit against the beats, and this
            # service transcribes none. The field stays in the payload so the
            # UI can consume one beat_grid shape from either backend.
            "onset_delay": grid.onset_delay,
        }
        if grid
        else None,
    }


@app.api_route("/health", methods=["GET", "HEAD"])
async def health() -> dict:
    """Liveness and readiness in one, deliberately cheap and never rate limited.

    The UI pings this before it uploads anything, to wake a Space that has gone
    to sleep — so it has to answer while the model is still loading, and has to
    stay outside the rate limiter or the ping would eat the client's budget.
    """
    return {
        "status": "ok" if _model_ready else "loading",
        "version": VERSION,
        "model": MODEL_NAME,
    }


@app.post("/analyze")
async def analyze(
    request: Request,
    file: Annotated[UploadFile, File()],
    # Same three-valued flag as /transcribe: "true" fails loudly when no tempo
    # is found, "false" doesn't even try, "best-effort" falls back to no grid.
    detect_tempo: Annotated[TempoDetection, Form()] = "best-effort",
) -> dict:
    """Recognize the chords in an uploaded recording.

    The gates run cheapest-first — secret, rate limit, byte cap, decode,
    duration cap — so that abuse is rejected before it costs anything, and only
    then is one of the few analysis slots taken.
    """
    _check_shared_secret(request)
    await _rate_limiter.check(_client_id(request))

    data = await _read_capped(file)
    wav, sample_rate = _decode(data, file.filename)
    duration = wav.shape[-1] / sample_rate
    if duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"audio is {duration:.0f}s long, the limit is {MAX_AUDIO_SECONDS}s",
        )

    if _analysis_slots.locked():
        # Queueing instead would just hand every waiting client a timeout; a
        # 503 lets the UI say "busy, try again" while it is still true.
        raise HTTPException(status_code=503, detail="all analysis slots are busy")

    wav = _to_mono_16k(wav, sample_rate)
    started = time.monotonic()
    await _analysis_slots.acquire()
    task = asyncio.create_task(asyncio.to_thread(_analyze_audio, wav, detect_tempo))
    # Released when the *work* finishes, not when this request stops waiting for
    # it: torch calls in a worker thread cannot be cancelled, so a timed-out
    # analysis is still burning a core. Freeing the slot at the timeout would
    # admit a second request onto the same busy CPU.
    task.add_done_callback(lambda _: _analysis_slots.release())
    try:
        result = await asyncio.wait_for(asyncio.shield(task), ANALYZE_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning("analysis timed out after %ds", ANALYZE_TIMEOUT_SEC)
        raise HTTPException(status_code=504, detail="analysis timed out") from None

    result["duration"] = duration
    result["processing_time_ms"] = int((time.monotonic() - started) * 1000)
    logger.info(
        "analyzed %.1fs of audio in %dms (%d chords)",
        duration,
        result["processing_time_ms"],
        len(result["chords"]),
    )
    return result
