import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { type Region } from "wavesurfer.js/dist/plugins/regions.esm.js";
import { Button } from "./Button";
import { cutRegion, decodeAudioFile, encodeWav, formatTime } from "../waveform";

/** The audio to transcribe, and the span of it to keep. */
export type Trim = { buffer: AudioBuffer; start: number; end: number };

/** A selection, or null for "all of the current audio". */
type Range = [number, number] | null;

/** One undoable state: the audio, and what is selected in it. */
type Snapshot = { buffer: AudioBuffer; range: Range };

/** Pulled from the piano-roll palette so the two views read as one app. */
const WAVE = "#5a5c68";
const PROGRESS = "#ff5b7a";
const REGION = "rgba(255, 91, 122, 0.16)";
const HEIGHT = 96;

/**
 * Waveform of the picked file, with a resizable selection you can transcribe
 * or delete.
 *
 * Transcription time scales with the length of the upload, so the useful edit
 * here is choosing *which* audio the model sees — keep the solo, or drop the
 * two minutes of applause — not mixing. Both edits stay client-side (see
 * `waveform.ts`): the file is decoded here to draw it, so the cut costs an
 * array copy, and the server keeps its single-file `/transcribe` contract.
 *
 * A file Web Audio cannot decode renders nothing and reports no trim — the
 * server may still read it through libsndfile, so this must never block a
 * transcription, only decline to offer the editor.
 */
export function WaveformEditor(props: {
  file: File;
  /** The audio to upload, or null when the untouched file should go up. */
  onChange: (trim: Trim | null) => void;
}) {
  const { file, onChange } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const waveRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const regionRef = useRef<Region | null>(null);
  /** The decoded file, kept to tell an edited buffer from an untouched one. */
  const originalRef = useRef<AudioBuffer | null>(null);

  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [range, setRange] = useState<Range>(null);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [zoom, setZoom] = useState(0);
  const [past, setPast] = useState<Snapshot[]>([]);
  const [future, setFuture] = useState<Snapshot[]>([]);
  // Mirror of `range`, read when recording a step without re-creating the
  // WaveSurfer event handlers on every selection change.
  const rangeRef = useRef<Range>(null);
  // Set while undo/redo rebuilds the region, so the plugin events that come
  // back from that don't record the restoration as a fresh edit.
  const restoringRef = useRef(false);
  // A selection to re-apply once a restored buffer has drawn: undoing a cut
  // swaps the audio, which builds a whole new waveform.
  const pendingRangeRef = useRef<Range>(null);

  // Held in a ref so a parent re-render doesn't tear the waveform down.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const duration = buffer?.duration ?? 0;
  const edited = buffer !== null && buffer !== originalRef.current;

  // The upload: the selection if there is one, the whole thing once it has been
  // cut, and null while the audio is still exactly what the user picked.
  useEffect(() => {
    if (buffer === null || (!edited && range === null)) {
      onChangeRef.current(null);
      return;
    }
    onChangeRef.current({
      buffer,
      start: range?.[0] ?? 0,
      end: range?.[1] ?? buffer.duration,
    });
  }, [buffer, range, edited]);

  function commitRange(next: Range) {
    rangeRef.current = next;
    setRange(next);
  }

  /** Record a settled edit, unless we are the ones replaying one. */
  function record(next: Snapshot) {
    if (restoringRef.current || buffer === null) return;
    // Read now, not inside the updater: React runs those during the next
    // render, by which point the refs have moved on and every step would
    // record itself as its own predecessor.
    const previous = { buffer, range: rangeRef.current };
    setPast((p) => [...p, previous]);
    setFuture([]);
    if (next.buffer !== buffer) setBuffer(next.buffer);
    commitRange(next.range);
  }

  /** Put a selection back on the current waveform without recording it. */
  function showRange(next: Range) {
    const regions = regionsRef.current;
    if (regions === null) return;
    restoringRef.current = true;
    try {
      if (next === null) {
        regionRef.current?.remove();
        regionRef.current = null;
      } else if (regionRef.current !== null) {
        regionRef.current.setOptions({ start: next[0], end: next[1] });
      } else {
        regionRef.current = regions.addRegion({
          start: next[0],
          end: next[1],
          color: REGION,
        });
      }
      commitRange(next);
    } finally {
      restoringRef.current = false;
    }
  }

  /** Restore `target`, rebuilding the waveform first if the audio differs. */
  function travel(target: Snapshot) {
    if (target.buffer === buffer) {
      showRange(target.range);
    } else {
      pendingRangeRef.current = target.range;
      commitRange(target.range);
      setBuffer(target.buffer);
    }
  }

  function undo() {
    if (past.length === 0 || buffer === null) return;
    const target = past[past.length - 1];
    const current = { buffer, range: rangeRef.current };
    setPast((p) => p.slice(0, -1));
    setFuture((f) => [current, ...f]);
    travel(target);
  }

  function redo() {
    if (future.length === 0 || buffer === null) return;
    const target = future[0];
    const current = { buffer, range: rangeRef.current };
    setFuture((f) => f.slice(1));
    setPast((p) => [...p, current]);
    travel(target);
  }

  /** Delete the selection; what is left of the track closes the gap. */
  function removeSelection() {
    if (buffer === null || range === null) return;
    record({ buffer: cutRegion(buffer, range[0], range[1]), range: null });
  }

  // Decoding is per file; everything below works off the decoded buffer, so an
  // edit doesn't re-read the upload.
  useEffect(() => {
    let cancelled = false;
    setFailed(false);
    setPast([]);
    setFuture([]);
    setBuffer(null);
    commitRange(null);
    originalRef.current = null;
    pendingRangeRef.current = null;
    decodeAudioFile(file)
      .then((decoded) => {
        if (cancelled) return;
        originalRef.current = decoded;
        setBuffer(decoded);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  // One WaveSurfer per buffer — a cut or an undo across one replaces the audio,
  // so the waveform is rebuilt rather than patched.
  useEffect(() => {
    if (containerRef.current === null || buffer === null) return;
    setPlaying(false);
    setZoom(0);
    let cancelled = false;

    const regions = RegionsPlugin.create();
    const wave = WaveSurfer.create({
      container: containerRef.current,
      waveColor: WAVE,
      progressColor: PROGRESS,
      cursorColor: PROGRESS,
      height: HEIGHT,
      normalize: true,
      plugins: [regions],
    });
    waveRef.current = wave;
    regionsRef.current = regions;

    wave.on("decode", () => {
      if (cancelled) return;
      // Regions created by dragging are draggable and resizable by default,
      // which is where the edge handles come from.
      regions.enableDragSelection({ color: REGION });
      // An undo landed on a state that had a selection: draw it now that there
      // is a waveform to draw it on.
      const pending = pendingRangeRef.current;
      pendingRangeRef.current = null;
      if (pending !== null) showRange(pending);
    });
    wave.on("error", () => setFailed(true));
    wave.on("play", () => setPlaying(true));
    wave.on("pause", () => setPlaying(false));
    wave.on("finish", () => setPlaying(false));

    // One selection at a time: a new drag replaces the old region rather than
    // stacking translucent boxes that are impossible to tell apart.
    regions.on("region-created", (r) => {
      if (cancelled) return;
      const previous = regionRef.current;
      regionRef.current = r;
      if (previous !== null && previous !== r) previous.remove();
      record({ buffer, range: [r.start, r.end] });
    });
    // Fires when a drag or resize *finishes* (`region-update` covers the
    // intermediate frames), so the trim only follows a settled gesture.
    regions.on("region-updated", (r) => {
      if (cancelled || r !== regionRef.current) return;
      record({ buffer, range: [r.start, r.end] });
    });
    // `wave.destroy()` removes the region on the way out, so this fires during
    // teardown too — `cancelled` keeps that from clearing a fresh state.
    regions.on("region-removed", (r) => {
      if (cancelled || r !== regionRef.current) return;
      regionRef.current = null;
      record({ buffer, range: null });
    });

    // Playback needs real audio behind the peaks. The picked file already is
    // that, until a cut makes the buffer something no file holds.
    const source =
      buffer === originalRef.current
        ? file
        : encodeWav(
            Array.from({ length: buffer.numberOfChannels }, (_, c) =>
              buffer.getChannelData(c),
            ),
            buffer.sampleRate,
          );
    const url = URL.createObjectURL(source);
    const peaks = Array.from({ length: buffer.numberOfChannels }, (_, c) =>
      buffer.getChannelData(c),
    );
    // Hand WaveSurfer the samples and the duration instead of letting it decode
    // the blob a second time. Not just an optimisation: given neither,
    // `loadAudio` waits on the <audio> element's `loadedmetadata`, which never
    // arrives for a blob src, and the waveform stays blank forever.
    wave.load(url, peaks, buffer.duration).catch(() => {
      if (!cancelled) setFailed(true);
    });

    return () => {
      cancelled = true;
      regionRef.current = null;
      wave.destroy();
      URL.revokeObjectURL(url);
      waveRef.current = null;
      regionsRef.current = null;
    };
    // `file` only feeds the playback URL, and it cannot change without the
    // decode effect above replacing the buffer too.
  }, [buffer]); // eslint-disable-line react-hooks/exhaustive-deps

  // WaveSurfer throws "No audio loaded" if asked to zoom before it holds the
  // decoded samples, which it doesn't yet when the `decode` event lands.
  useEffect(() => {
    const wave = waveRef.current;
    if (wave !== null && wave.getDecodedData() !== null) wave.zoom(zoom);
  }, [zoom, buffer]);

  /** Play the selection (stopping at its end), or all of it without one. */
  function playPause() {
    const wave = waveRef.current;
    if (wave === null) return;
    if (wave.isPlaying()) wave.pause();
    else if (regionRef.current !== null) regionRef.current.play(true);
    else wave.play();
  }

  if (failed) return null;

  // Cutting everything away would leave an empty buffer, which is not audio.
  const wholeTrackSelected =
    range !== null && range[1] - range[0] >= duration - 0.001;

  return (
    <section className="card flex flex-col gap-2 p-4">
      <div ref={containerRef} className="w-full" />

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={playPause} disabled={duration === 0}>
          {playing ? "Pause" : range === null ? "Play" : "Play selection"}
        </Button>
        <div className="flex items-center gap-1">
          <Button onClick={undo} disabled={past.length === 0} aria-label="Undo">
            Undo
          </Button>
          <Button onClick={redo} disabled={future.length === 0} aria-label="Redo">
            Redo
          </Button>
        </div>
        <Button
          onClick={removeSelection}
          disabled={range === null || wholeTrackSelected}
          title={
            wholeTrackSelected
              ? "That would leave nothing to transcribe"
              : "Delete the selected audio"
          }
        >
          Remove selection
        </Button>
        <label className="flex items-center gap-2 text-sm text-muted">
          Zoom
          <input
            type="range"
            min={0}
            max={200}
            value={zoom}
            disabled={duration === 0}
            onChange={(e) => setZoom(Number(e.target.value))}
            className="w-28"
          />
        </label>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted">
        <span>
          {duration === 0 ? (
            "Reading the audio…"
          ) : range === null ? (
            <>
              {edited ? "Edited down to" : "Whole track,"} {formatTime(duration)}.
              Drag to select a part.
            </>
          ) : (
            <>
              <span className="text-content">
                {formatTime(range[0])} – {formatTime(range[1])}
              </span>{" "}
              of {formatTime(duration)}
            </>
          )}
        </span>
        {range !== null && (
          <Button
            kind="ghost"
            className="px-1 py-0.5 text-sm text-muted underline underline-offset-4 hover:bg-transparent enabled:hover:text-content"
            onClick={() => regionRef.current?.remove()}
          >
            Clear selection
          </Button>
        )}
      </div>
    </section>
  );
}
