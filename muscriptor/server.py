"""FastAPI server exposing transcription as an SSE event stream.

POST /transcribe with an audio file (multipart/form-data field `file`; WAV,
or any format soundfile/libsndfile can read — mp3, flac, ogg, m4a, …) returns
`text/event-stream`. Each event's data is a JSON dict tagged by `type`:
`queued` position updates while other requests are ahead,
`transcription_started` when this request reaches the front, `start` / `end`
note events (same shape as `muscriptor.main._event_to_dict`), `progress` chunk
anchors (`{completed, total}`), and a final
`transcription_complete` event carrying the base64-encoded .mid file (`data`)
plus the detected `beat_grid`
(`{bpm, beats_per_bar, first_downbeat, onset_delay}`, or null if no tempo was
found) and the recognized `chords` (`[{time, label, root, intervals}]`, chord
changes in the same times the beat grid is drawn in, each with the notes it is
made of so a client can sound it). `onset_delay` is how late the streamed
note times are against those beats: the MIDI has it taken out already, an SSE
consumer has to subtract it. The chord times need no such correction — they
were snapped to the beats themselves.

POST /transcribe/midi takes the same upload but blocks until transcription
completes and returns the raw `audio/midi` bytes directly (no SSE, no
base64), with a `Content-Disposition: attachment` header. Audio longer than
15 minutes is rejected with 413.

POST /sheets takes a MIDI upload instead of audio (the `quantized_midi` from
/transcribe, with `quantized=true`) and returns every file
`muscriptor.utils.sheets.write_sheets` engraves from it — MusicXML, the full
score, one PDF per instrument — as a single uncompressed zip. It needs
MuseScore 4+ on the server, and answers 503 when there is none.
"""

import asyncio
import base64
import dataclasses
import io
import json
import os
import tempfile
import threading
import time
import wave
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.soundfonts import SF3_URL
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.audio import _read_non_wav_file, _read_wav_file
from muscriptor.utils.beats import BeatDetectionError, TempoDetection
from muscriptor.utils.chords import prefers_flats, published_chords
from muscriptor.utils.download import download_if_necessary
from muscriptor.utils.sheets import (
    MuseScoreError,
    MuseScoreNotFoundError,
    write_sheets,
)


@dataclasses.dataclass(eq=False)
class _TranscriptionTicket:
    client_id: str | None
    cancellable: bool
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event)


class _TranscriptionQueue:
    """A process-local FIFO for the one model loaded by ``muscriptor serve``.

    Streaming requests get position updates while waiting. A newer request from
    the same browser tab cancels the tab's active/queued request, but it joins
    the back of the line so repeated submissions cannot jump other users.
    """

    def __init__(self, heartbeat_s: float = 15.0):
        self._condition = threading.Condition()
        self._active: _TranscriptionTicket | None = None
        self._waiting: list[_TranscriptionTicket] = []
        self._heartbeat_s = heartbeat_s

    @property
    def waiting_count(self) -> int:
        with self._condition:
            return len(self._waiting)

    def submit(
        self, client_id: str | None, *, cancellable: bool
    ) -> _TranscriptionTicket:
        ticket = _TranscriptionTicket(client_id, cancellable)
        with self._condition:
            if client_id is not None:
                if (
                    self._active is not None
                    and self._active.client_id == client_id
                    and self._active.cancellable
                ):
                    self._active.cancel.set()

                kept: list[_TranscriptionTicket] = []
                for waiting in self._waiting:
                    if waiting.client_id == client_id:
                        waiting.cancel.set()
                    else:
                        kept.append(waiting)
                self._waiting = kept

            if self._active is None:
                self._active = ticket
            else:
                self._waiting.append(ticket)
            self._condition.notify_all()
        return ticket

    def wait_for_update(
        self, ticket: _TranscriptionTicket, last_position: int | None
    ) -> tuple[str, int | None]:
        """Block until a ticket starts, is cancelled, or needs an SSE update."""
        deadline = time.monotonic() + self._heartbeat_s
        with self._condition:
            while True:
                if ticket.cancel.is_set():
                    return "cancelled", None
                if self._active is ticket:
                    return "active", 0
                try:
                    # The active transcription plus earlier waiters are ahead.
                    position = self._waiting.index(ticket) + 1
                except ValueError:
                    return "cancelled", None
                if position != last_position:
                    return "queued", position

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Re-send the current position as an SSE heartbeat so long
                    # waits stay alive through reverse proxies.
                    return "queued", position
                self._condition.wait(remaining)

    def wait_until_active(self, ticket: _TranscriptionTicket) -> bool:
        """Wait for a non-streaming request's turn."""
        with self._condition:
            while True:
                if ticket.cancel.is_set():
                    return False
                if self._active is ticket:
                    return True
                if ticket not in self._waiting:
                    return False
                self._condition.wait()

    def cancel(self, ticket: _TranscriptionTicket) -> None:
        """Cancel a queued ticket, or signal a cancellable active one."""
        with self._condition:
            if self._active is ticket:
                if ticket.cancellable:
                    ticket.cancel.set()
            elif ticket in self._waiting:
                self._waiting.remove(ticket)
                ticket.cancel.set()
            else:
                return
            self._condition.notify_all()

    def finish(self, ticket: _TranscriptionTicket) -> None:
        """Remove a ticket and promote the oldest live waiter."""
        with self._condition:
            if self._active is ticket:
                self._active = None
            elif ticket in self._waiting:
                self._waiting.remove(ticket)
                ticket.cancel.set()
            else:
                return

            while self._waiting:
                candidate = self._waiting.pop(0)
                if not candidate.cancel.is_set():
                    self._active = candidate
                    break
            self._condition.notify_all()


_MAX_TRANSCRIBE_MIDI_DURATION_S = 15 * 60

SHEETS_ZIP_NAME = "sheets.zip"


def _allowed_origins() -> list[str]:
    """Browser origins allowed to call a standalone MuScriptor API.

    The default remains same-origin only: without this environment variable no
    CORS middleware is installed, preserving the bundled UI deployment.  A
    standalone frontend can opt in with a comma-separated list of exact origins.
    """
    return [
        origin.strip().rstrip("/")
        for origin in os.environ.get("MUSCRIPTOR_ALLOWED_ORIGINS", "").split(",")
        if origin.strip() and origin.strip() != "*"
    ]


def engrave_to_zip(midi_bytes: bytes, quantized: bool = False) -> bytes:
    """Engrave `midi_bytes` and pack everything written into one zip.

    Runs `write_sheets` into a scratch directory that is thrown away once the
    archive is built, so the server keeps nothing on disk between requests.
    Member names are the bare filenames — the directory layout documented under
    "Sheet music" in the README, flattened by one level.

    Stored, not deflated: the client unpacks this archive in the browser to
    offer the files one at a time, and all but the MusicXML are PDFs, which
    carry compression of their own.
    """
    with tempfile.TemporaryDirectory(prefix="muscriptor-sheets-") as tmp:
        written = write_sheets(midi_bytes, Path(tmp), quantized=quantized)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in written:
                archive.write(path, arcname=path.name)
        return buf.getvalue()


def event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


def create_app(model: TranscriptionModel, web_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="muscriptor")
    allowed_origins = _allowed_origins()
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-Client-Id"],
        )

    transcription_queue = _TranscriptionQueue()
    app.state.transcription_queue = transcription_queue

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/instruments")
    async def list_instruments():
        return {"instruments": list(MT3_FULL_PLUS_GROUP_NAMES.keys())}

    @app.get("/soundfonts/MuseScore_General.sf3")
    async def soundfont() -> FileResponse:
        """Compressed soundfont for the web UI's in-browser synthesizer.

        Fetched from SF3_URL on first request (in a worker thread, so the
        event loop keeps serving) and cached locally.
        """
        path = await asyncio.to_thread(download_if_necessary, SF3_URL)
        return FileResponse(path, media_type="application/octet-stream")

    @app.post("/transcribe")
    async def transcribe(
        file: Annotated[UploadFile, File()],
        instruments: Annotated[list[str], Form(default_factory=list)],
        # "true" fails loudly on tempo detection errors, "false" doesn't even try
        detect_tempo: Annotated[TempoDetection, Form()] = "best-effort",
        # Chord recognition costs a few seconds of CPU on top of the tempo
        # detection; a client that only wants notes can turn it off.
        chords: Annotated[bool, Form()] = True,
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        data = await file.read()
        # PCM WAV goes through the stdlib reader (keeps WAV decoding byte-for-byte
        # identical to the CLI); anything that isn't a readable WAV (mp3, flac,
        # ogg, m4a, …) falls back to soundfile/libsndfile. A genuinely
        # undecodable upload (corrupt/truncated file, or a format libsndfile
        # can't read) is the client's fault, so report it as a 400 rather than
        # letting it surface as a 500.
        try:
            wav, sr = _read_wav_file(io.BytesIO(data))
        except (wave.Error, EOFError):
            try:
                wav, sr = _read_non_wav_file(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"could not decode audio file '{file.filename}': {e}",
                ) from e

        unknown = [n for n in instruments if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )

        ticket = transcription_queue.submit(x_client_id, cancellable=True)

        def gen():
            try:
                last_position: int | None = None
                while True:
                    state, position = transcription_queue.wait_for_update(
                        ticket, last_position
                    )
                    if state == "cancelled":
                        return
                    if state == "active":
                        break
                    last_position = position
                    yield f"data: {json.dumps({'type': 'queued', 'position': position})}\n\n"

                # The frontend stays on the upload screen through queue updates
                # and moves to the transcription view only after this event.
                yield 'data: {"type": "transcription_started"}\n\n'
                events: list[NoteStartEvent | NoteEndEvent] = []
                # batch_size=1 so each chunk's notes stream out as soon as it is
                # generated, instead of waiting for a whole batch of chunks.
                # no_eos_is_ok=True so one runaway chunk that never emits EOS only
                # warns (and keeps its notes) instead of aborting the whole stream.
                for ev in model.transcribe(
                    (wav, sr),
                    instruments=instruments or None,
                    batch_size=1,
                    no_eos_is_ok=True,
                ):
                    # A newer request superseded this run — stop generating and
                    # release its queue slot, at most one chunk after the signal.
                    if ticket.cancel.is_set():
                        return
                    if isinstance(ev, ProgressEvent):
                        # Coarse chunk-completion anchor — forward it but keep it
                        # out of the note list the MIDI file is built from.
                        payload = json.dumps(
                            {
                                "type": "progress",
                                "completed": ev.completed,
                                "total": ev.total,
                            }
                        )
                        yield f"data: {payload}\n\n"
                        continue
                    events.append(ev)
                    payload = json.dumps(event_to_dict(ev))
                    yield f"data: {payload}\n\n"
                # All notes streamed — build the MIDI file in memory (reusing the
                # exact `muscriptor transcribe` logic) and send it as a final event
                # with the bytes base64-encoded.
                if ticket.cancel.is_set():
                    return
                # Detect tempo/meter only now: it costs a few seconds of CPU and
                # nothing before this point needs it, so the notes stream first.
                grid = model.detect_beat_grid_for((wav, sr), detect_tempo)
                # Measure the onset lag up here rather than leaving it to the MIDI
                # writing, since the UI has to be told the very same number to move
                # the notes it already drew.
                if grid:
                    grid = grid.with_onset_delay(
                        [
                            ev.start_time
                            for ev in events
                            if isinstance(ev, NoteStartEvent)
                        ]
                    )
                # Chords come from the audio, not the notes, so they are
                # recognized here rather than derived from the event stream —
                # and snapped to the grid that was just detected, which is what
                # puts a chord change on a bar line instead of near one.
                recognized = (
                    model.recognize_chords_for((wav, sr), grid) if chords else []
                )
                midi_bytes = model.events_to_midi_bytes(
                    iter(events), beat_grid=grid, chords=recognized
                )
                # Decided once, so the symbols the UI shows are spelled exactly
                # like the ones just written into the MIDI.
                spelling = prefers_flats(recognized)
                midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
                # A second copy with the notes snapped to the beat grid. Useful for
                # writing sheet music where we want "idealized" timing
                quantized_midi = (
                    model.events_to_midi_bytes(
                        iter(events),
                        beat_grid=grid,
                        chords=recognized,
                        quantize=True,
                    )
                    if grid is not None and grid.beat_subdivision is not None
                    else None
                )
                # The grid rides along so the UI can draw bar lines instead of a
                # fixed seconds grid; null when no tempo was detected.
                payload = json.dumps(
                    {
                        "type": "transcription_complete",
                        "data": midi_b64,
                        "quantized_midi": base64.b64encode(quantized_midi).decode(
                            "ascii"
                        )
                        if quantized_midi
                        else None,
                        # Only the fields the UI draws with; `grid.beats` is an
                        # ndarray and not JSON-serializable anyway.
                        "beat_grid": {
                            "bpm": grid.bpm,
                            "beats_per_bar": grid.beats_per_bar,
                            "first_downbeat": grid.first_downbeat,
                            # Seconds the streamed note times sit late against
                            # the beats; the MIDI already has it taken out, the
                            # UI has to subtract it from the notes it drew.
                            "onset_delay": grid.onset_delay,
                        }
                        if grid
                        else None,
                        # The chord track, in the same times the beat grid is
                        # drawn in (no onset_delay: the chords were snapped to
                        # the beats, not to the notes). Same symbols, and the
                        # same spelling, as the MIDI markers carry. `root` and
                        # `intervals` are what the chord is made of, so the UI
                        # can sound it without knowing any music theory; both
                        # are empty for an "N.C." span.
                        "chords": [
                            {
                                "time": chord.start,
                                "label": chord.label(spelling),
                                "root": chord.root,
                                "intervals": list(chord.intervals),
                            }
                            for chord in published_chords(recognized)
                        ],
                    }
                )
                yield f"data: {payload}\n\n"
            finally:
                transcription_queue.finish(ticket)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            # `finish` is idempotent with the generator's finally and also
            # removes a ticket if the client disconnects before first iteration.
            background=BackgroundTask(transcription_queue.finish, ticket),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/transcribe/midi")
    async def transcribe_midi(
        file: Annotated[UploadFile, File()],
        instruments: Annotated[list[str], Form(default_factory=list)],
        # "true" fails loudly on tempo detection errors, "false" doesn't even try
        detect_tempo: Annotated[TempoDetection, Form()] = "best-effort",
        chords: Annotated[bool, Form()] = True,
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Transcribe an audio file and return the .mid file directly.

        Unlike /transcribe, this blocks until transcription finishes and
        returns the raw MIDI bytes (no SSE, no base64) with a
        Content-Disposition header, so a plain HTTP client can save the
        response straight to disk.
        """
        data = await file.read()
        try:
            wav, sr = _read_wav_file(io.BytesIO(data))
        except (wave.Error, EOFError):
            try:
                wav, sr = _read_non_wav_file(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"could not decode audio file '{file.filename}': {e}",
                ) from e

        duration_s = wav.shape[-1] / sr
        if duration_s > _MAX_TRANSCRIBE_MIDI_DURATION_S:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"audio file is {duration_s / 60:.1f} minutes long; "
                    f"the limit is {_MAX_TRANSCRIBE_MIDI_DURATION_S // 60:.0f} minutes"
                ),
            )

        unknown = [n for n in instruments if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )

        # This blocking endpoint shares the same FIFO as the SSE endpoint. Once
        # it starts it is not cancellable mid-model-call, so another request can
        # never promote early and run on the GPU at the same time.
        ticket = transcription_queue.submit(x_client_id, cancellable=False)
        try:
            if not await asyncio.to_thread(
                transcription_queue.wait_until_active, ticket
            ):
                raise HTTPException(status_code=409, detail="request was superseded")

            work = asyncio.create_task(
                asyncio.to_thread(
                    model.transcribe_and_postprocess,
                    (wav, sr),
                    instruments=instruments or None,
                    detect_tempo=detect_tempo,
                    recognize_chords=chords,
                )
            )
            try:
                midi_bytes, _ = await asyncio.shield(work)
            except asyncio.CancelledError:
                # The thread cannot be interrupted safely. Keep this queue slot
                # until it exits so the next ticket never overlaps it on the GPU.
                await work
                raise
        except BeatDetectionError as e:
            # Only reachable with detect_tempo=true, where the caller wants an error
            # if tempo detection fails.
            raise HTTPException(status_code=422, detail=str(e)) from e
        finally:
            transcription_queue.finish(ticket)

        return Response(
            content=midi_bytes,
            media_type="audio/midi",
            headers={"Content-Disposition": 'attachment; filename="result.mid"'},
        )

    @app.post("/auralize")
    async def auralize(
        midi: Annotated[UploadFile, File()],
        audio: Annotated[UploadFile | None, File()] = None,
        mode: Annotated[str, Form()] = "mix",
    ):
        """Render a transcription as WAV.

        mode="mix": stereo, original audio (L) + FluidSynth synthesis (R);
        requires the `audio` upload. mode="synth": mono, just the synthesis.
        """
        from muscriptor.utils.auralization import auralize as do_auralize
        from muscriptor.utils.auralization import synthesize

        if mode not in ("mix", "synth"):
            raise HTTPException(status_code=400, detail=f"unknown mode: {mode!r}")
        if mode == "mix" and audio is None:
            raise HTTPException(
                status_code=400, detail="mode='mix' requires an audio file"
            )

        midi_data = await midi.read()
        tmp_paths: list[str] = []

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
            tmp_midi.write(midi_data)
            midi_tmp = tmp_midi.name
            tmp_paths.append(midi_tmp)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_tmp = tmp_out.name
            tmp_paths.append(out_tmp)

        try:
            if mode == "synth":
                synthesize(midi_path=midi_tmp, output_path=out_tmp)
            else:
                audio_data = await audio.read()
                suffix = Path(audio.filename or "audio.wav").suffix.lower() or ".wav"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as tmp_audio:
                    tmp_audio.write(audio_data)
                    tmp_paths.append(tmp_audio.name)
                do_auralize(
                    midi_path=midi_tmp,
                    original_audio_path=tmp_audio.name,
                    output_path=out_tmp,
                )
            with open(out_tmp, "rb") as f:
                wav_bytes = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.unlink(p)

        return Response(content=wav_bytes, media_type="audio/wav")

    @app.post("/sheets")
    async def sheets(
        midi: Annotated[UploadFile, File()],
        quantized: Annotated[bool, Form()] = False,
    ) -> Response:
        """Engrave a MIDI file as sheet music, returned as one zip.

        The whole set is rendered in one go — MuseScore is slow enough that a
        round trip per file would be worse — so the caller gets every PDF, the
        MusicXML and the MIDI in a single uncompressed archive and picks from it
        locally. Requires MuseScore 4+ on the server (503 without it).

        `quantized` says the upload is already snapped to a beat grid — the
        `quantized_midi` from /transcribe — which is what the notation should be
        engraved from. Without it the engraving keeps the timing jitter, so this
        does not quantize anything itself.
        """
        midi_bytes = await midi.read()

        try:
            zip_bytes = await asyncio.to_thread(engrave_to_zip, midi_bytes, quantized)
        except MuseScoreNotFoundError as e:
            # A deployment problem, not a bad request: the same 503 the UI
            # already knows how to report, with the install hint as its detail.
            raise HTTPException(status_code=503, detail=str(e)) from e
        except MuseScoreError as e:
            # MuseScore ran but wrote nothing usable — most often because the
            # upload wasn't a MIDI file it could import.
            raise HTTPException(status_code=500, detail=str(e)) from e

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{SHEETS_ZIP_NAME}"'
            },
        )

    if web_dir is not None:
        web_path = Path(web_dir)
        if web_path.is_dir():
            app.mount("/", StaticFiles(directory=web_path, html=True), name="web")

    return app
