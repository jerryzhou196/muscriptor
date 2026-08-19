import { useEffect, useState } from "react";
import { Button } from "./Button";
import { Waveform, useWaveform } from "./Waveform";
import {
  copyRegion,
  cutRegion,
  decodeAudioFile,
  formatTime,
  trimToWavFile,
} from "../waveform";

type Range = [number, number] | null;

export function WaveformEditor(props: {
  file: File;
  onChange: (file: File) => void;
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
  const canEdit = selected > 0.01 && selected < duration - 0.01;

  useEffect(() => {
    if (buffer === null || (!edited && range === null)) {
      onChange(file);
      return;
    }
    onChange(
      trimToWavFile(
        file,
        buffer,
        range?.[0] ?? 0,
        range?.[1] ?? duration,
      ),
    );
  }, [buffer, range, edited, duration, file, onChange]);

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

  function showAudio(next: AudioBuffer) {
    pause();
    setRange(null);
    setBuffer(next);
  }

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
