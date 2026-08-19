import { useEffect, useState } from "react";
import { Button } from "./Button";
import { Waveform, useWaveform } from "./Waveform";
import {
  copyRegion,
  cutRegion,
  decodeAudioFile,
  formatTime,
} from "../waveform";

/** The audio to transcribe, and the span of it to keep. */
export type Trim = { buffer: AudioBuffer; start: number; end: number };

/** A selection, or null for "all of the current audio". */
type Range = [number, number] | null;

/**
 * Waveform of the picked file, with a selection to play, cut, or crop to.
 *
 * Modelled on AudioMass: the decoded AudioBuffer is the document, and every
 * edit builds a new buffer and reloads the view from it. Previous buffers stay
 * in local history so Command/Control-Z can undo without re-decoding the file.
 * Drawing and playback live in `useWaveform`.
 *
 * A file Web Audio cannot decode renders nothing and reports no trim. The
 * server may still read it, so this must never block a transcription, only
 * decline to offer the editor.
 */
export function WaveformEditor(props: {
  file: File;
  /** The audio to upload, or null when the untouched file should go up. */
  onChange: (trim: Trim | null) => void;
}) {
  const { file, onChange } = props;

  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [original, setOriginal] = useState<AudioBuffer | null>(null);
  const [past, setPast] = useState<AudioBuffer[]>([]);
  const [future, setFuture] = useState<AudioBuffer[]>([]);
  const [range, setRange] = useState<Range>(null);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);

  const duration = buffer?.duration ?? 0;
  const edited = buffer !== null && buffer !== original;
  const { containerRef, playPause, pause } = useWaveform({
    file,
    buffer,
    edited,
    range,
    onRangeChange: setRange,
    onPlayingChange: setPlaying,
    onError: () => setFailed(true),
    active: !failed,
  });
  const selected = range === null ? 0 : range[1] - range[0];
  // An edit may consume neither the whole track nor none of it: a zero-length
  // AudioBuffer is invalid. The tolerance covers a drag landing a sample short.
  const canEdit = selected > 0.01 && selected < duration - 0.01;

  // The upload: the selection if there is one, the whole buffer once edited,
  // and null while the audio is still exactly what the user picked.
  useEffect(() => {
    if (buffer === null || (!edited && range === null)) onChange(null);
    else if (range === null) onChange({ buffer, start: 0, end: duration });
    else onChange({ buffer, start: range[0], end: range[1] });
  }, [buffer, range, edited, duration, onChange]);

  // Decode once per file; everything below works off the buffer.
  useEffect(() => {
    let cancelled = false;
    setFailed(false);
    setPast([]);
    setFuture([]);
    setBuffer(null);
    setRange(null);
    setOriginal(null);
    decodeAudioFile(file)
      .then((decoded) => {
        if (cancelled) return;
        setOriginal(decoded);
        setBuffer(decoded);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  /** Show an audio state without recording how we got there. */
  function showAudio(next: AudioBuffer) {
    pause();
    setRange(null);
    setBuffer(next);
  }

  /** Swap in an edit and discard any redo path from the current state. */
  function replaceAudio(next: AudioBuffer) {
    if (buffer === null) return;
    setPast((p) => [...p, buffer]);
    setFuture([]);
    showAudio(next);
  }

  function undo() {
    if (buffer === null || past.length === 0) return;
    const prev = past[past.length - 1];
    setFuture((f) => [...f, buffer]);
    setPast((p) => p.slice(0, -1));
    showAudio(prev);
  }

  function redo() {
    if (buffer === null || future.length === 0) return;
    const next = future[future.length - 1];
    setPast((p) => [...p, buffer]);
    setFuture((f) => f.slice(0, -1));
    showAudio(next);
  }

  function cut() {
    if (buffer !== null && range !== null) {
      replaceAudio(cutRegion(buffer, range[0], range[1]));
    }
  }

  function crop() {
    if (buffer !== null && range !== null) {
      replaceAudio(copyRegion(buffer, range[0], range[1]));
    }
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        (!event.metaKey && !event.ctrlKey) ||
        event.altKey ||
        event.key.toLowerCase() !== "z"
      ) {
        return;
      }
      const target = event.target as HTMLElement | null;
      if (
        target?.isContentEditable ||
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.tagName === "SELECT"
      ) {
        return;
      }
      event.preventDefault();
      if (event.shiftKey) redo();
      else undo();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [buffer, past, future]);

  if (failed) return null;

  return (
    <section className="card flex flex-col gap-2 p-4">
      <Waveform ref={containerRef} />

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={playPause} disabled={duration === 0}>
          {playing ? "Pause" : range === null ? "Play" : "Play selection"}
        </Button>
        <Button
          onClick={cut}
          disabled={!canEdit}
          title="Delete the selection and close the gap"
        >
          Cut selection
        </Button>
        <Button
          onClick={crop}
          disabled={!canEdit}
          title="Keep only the selection"
        >
          Crop to selection
        </Button>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted">
        <span>
          {duration === 0 ? (
            "Reading the audio…"
          ) : range === null ? (
            <>
              {edited ? "Edited audio" : "Whole track"}, {formatTime(duration)}.
              Drag to select a part, scroll to zoom.
            </>
          ) : (
            <>
              <span className="text-content">
                {formatTime(range[0])} – {formatTime(range[1])}
              </span>{" "}
              of {formatTime(duration)}. Drag it to move it.
            </>
          )}
        </span>
        {range !== null && (
          <Button
            kind="ghost"
            className="px-1 py-0.5 text-sm text-muted underline underline-offset-4 hover:bg-transparent enabled:hover:text-content"
            onClick={() => setRange(null)}
          >
            Clear selection
          </Button>
        )}
      </div>
    </section>
  );
}
