import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { type Region } from "wavesurfer.js/dist/plugins/regions.esm.js";
import { Button } from "./Button";
import { decodeAudioFile, formatTime } from "../waveform";

/** The span of `buffer` to transcribe, in seconds. */
export type Trim = { buffer: AudioBuffer; start: number; end: number };

/** A selection, or null for "the whole file". */
type Range = [number, number] | null;

/** Pulled from the piano-roll palette so the two views read as one app. */
const WAVE = "#5a5c68";
const PROGRESS = "#ff5b7a";
const REGION = "rgba(255, 91, 122, 0.16)";
const HEIGHT = 96;

/**
 * Waveform of the picked file, with a drag-to-select region you can resize.
 *
 * Transcription time scales with the length of the upload, so the common wish
 * — "just the solo, not the whole album side" — is answered here rather than by
 * making the user cut the file in another program first. The trim itself
 * happens client-side at submit (see `waveform.ts`): the file has to be decoded
 * to draw it anyway, and the server keeps its single-file `/transcribe`
 * contract.
 *
 * A file Web Audio cannot decode renders nothing and reports no trim — the
 * server may still read it through libsndfile, so this must never block a
 * transcription, only decline to offer the editor.
 */
export function WaveformEditor(props: {
  file: File;
  /** The selected span, or null when the whole (untouched) file should go up. */
  onChange: (trim: Trim | null) => void;
}) {
  const { file, onChange } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const waveRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const regionRef = useRef<Region | null>(null);
  const bufferRef = useRef<AudioBuffer | null>(null);

  const [duration, setDuration] = useState(0);
  const [range, setRange] = useState<Range>(null);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [zoom, setZoom] = useState(0);
  // Undo/redo over the selection alone — the file is never modified here, so
  // one range per step is the whole history.
  const [past, setPast] = useState<Range[]>([]);
  const [future, setFuture] = useState<Range[]>([]);
  // Mirror of `range`, read when recording a step without re-creating the
  // WaveSurfer event handlers on every selection change.
  const rangeRef = useRef<Range>(null);
  // Set while undo/redo rebuilds the region, so the plugin events that come
  // back from that don't record the restoration as a fresh edit.
  const restoringRef = useRef(false);

  // Held in a ref so a parent re-render doesn't tear the waveform down.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    onChangeRef.current(
      bufferRef.current && range
        ? { buffer: bufferRef.current, start: range[0], end: range[1] }
        : null,
    );
  }, [range]);

  function commitRange(next: Range) {
    rangeRef.current = next;
    setRange(next);
  }

  /** Record a settled edit, unless we are the ones replaying one. */
  function record(next: Range) {
    if (restoringRef.current) return;
    // Read now, not inside the updater: React runs those during the next
    // render, by which point `commitRange` has moved the ref on and every step
    // would record itself as its own predecessor.
    const previous = rangeRef.current;
    setPast((p) => [...p, previous]);
    setFuture([]);
    commitRange(next);
  }

  /** Put `next` back on the waveform without recording it as an edit. */
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

  function undo() {
    if (past.length === 0) return;
    const previous = past[past.length - 1];
    const current = rangeRef.current;
    setPast((p) => p.slice(0, -1));
    setFuture((f) => [current, ...f]);
    showRange(previous);
  }

  function redo() {
    if (future.length === 0) return;
    const next = future[0];
    const current = rangeRef.current;
    setFuture((f) => f.slice(1));
    setPast((p) => [...p, current]);
    showRange(next);
  }

  // One WaveSurfer per file. `file` is the only dependency: everything else the
  // effect touches is a ref, so a render can't restart the decode.
  useEffect(() => {
    if (containerRef.current === null) return;
    setFailed(false);
    setDuration(0);
    setPlaying(false);
    setZoom(0);
    setPast([]);
    setFuture([]);
    commitRange(null);
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

    wave.on("decode", (seconds) => {
      if (cancelled) return;
      setDuration(seconds);
      // Regions created by dragging are draggable and resizable by default,
      // which is where the edge handles come from.
      regions.enableDragSelection({ color: REGION });
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
      record([r.start, r.end]);
    });
    // Fires when a drag or resize *finishes* (`region-update` covers the
    // intermediate frames), so the trim only follows a settled gesture.
    regions.on("region-updated", (r) => {
      if (cancelled || r !== regionRef.current) return;
      record([r.start, r.end]);
    });
    // `wave.destroy()` removes the region on the way out, so this fires during
    // teardown too — `cancelled` keeps that from clearing a fresh file's state.
    regions.on("region-removed", (r) => {
      if (cancelled || r !== regionRef.current) return;
      regionRef.current = null;
      record(null);
    });

    const url = URL.createObjectURL(file);

    // Decode first, then hand WaveSurfer the samples and the duration instead
    // of letting it fetch and decode the file a second time. Not just an
    // optimisation: given neither, `loadAudio` waits on the <audio> element's
    // `loadedmetadata`, which never arrives for a blob src, and the waveform
    // stays blank forever. We need the AudioBuffer for the trim anyway.
    decodeAudioFile(file)
      .then((buf) => {
        if (cancelled) return;
        bufferRef.current = buf;
        const peaks = Array.from({ length: buf.numberOfChannels }, (_, c) =>
          buf.getChannelData(c),
        );
        return wave.load(url, peaks, buf.duration);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });

    return () => {
      cancelled = true;
      wave.destroy();
      URL.revokeObjectURL(url);
      waveRef.current = null;
      regionsRef.current = null;
      regionRef.current = null;
      bufferRef.current = null;
    };
  }, [file]);

  // WaveSurfer throws "No audio loaded" if asked to zoom before it holds the
  // decoded samples, which it doesn't yet when the `decode` event lands.
  useEffect(() => {
    const wave = waveRef.current;
    if (wave !== null && wave.getDecodedData() !== null) wave.zoom(zoom);
  }, [zoom, duration]);

  /** Play the selection (stopping at its end), or the whole file without one. */
  function playPause() {
    const wave = waveRef.current;
    if (wave === null) return;
    if (wave.isPlaying()) wave.pause();
    else if (regionRef.current !== null) regionRef.current.play(true);
    else wave.play();
  }

  if (failed) return null;

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
            <>Whole track, {formatTime(duration)}. Drag to transcribe only a part.</>
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
            Select the whole track
          </Button>
        )}
      </div>
    </section>
  );
}
