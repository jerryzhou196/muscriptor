import { useEffect, useRef, useState } from "react";
import { Button } from "./Button";
import { decodeAudioFile, formatTime, peaks } from "../waveform";

/** The span of `buffer` to transcribe, in seconds. */
export type Trim = { buffer: AudioBuffer; start: number; end: number };

const HEIGHT = 96;
/** Shorter drags are a click that missed, not a selection. */
const MIN_SELECTION_SECONDS = 0.2;

const COLORS = {
  wave: "rgba(237, 236, 240, 0.28)",
  waveSelected: "#ff5b7a",
  edge: "rgba(255, 91, 122, 0.65)",
  selectedBg: "rgba(255, 91, 122, 0.09)",
};

/**
 * Waveform of the picked file, with a drag-to-select region.
 *
 * Transcription time scales with the length of the upload, so the common wish
 * — "just the solo, not the whole album side" — is answered here rather than by
 * making the user cut the file in another program first. The trim happens
 * client-side (see `waveform.ts`): the file has to be decoded to draw it
 * anyway, and the server keeps its single-file `/transcribe` contract.
 *
 * A file the browser cannot decode (a codec Web Audio lacks) renders nothing
 * and reports no trim — the server may still handle it through libsndfile, so
 * this must never block a transcription, only decline to offer the editor.
 */
export function WaveformEditor(props: {
  file: File;
  /** The selected span, or null when the whole (untouched) file should go up. */
  onChange: (trim: Trim | null) => void;
}) {
  const { file, onChange } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [failed, setFailed] = useState(false);
  // null = the whole file. Kept in seconds so it survives a resize.
  const [region, setRegion] = useState<{ start: number; end: number } | null>(null);

  // Called on every region change; held in a ref so decoding doesn't restart
  // when the parent re-renders with a fresh closure.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    let cancelled = false;
    setBuffer(null);
    setFailed(false);
    setRegion(null);
    decodeAudioFile(file)
      .then((b) => !cancelled && setBuffer(b))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [file]);

  useEffect(() => {
    onChangeRef.current(buffer && region ? { buffer, ...region } : null);
  }, [buffer, region]);

  // Redraw on buffer/region change and on resize. The peaks are recomputed with
  // the canvas: one value per device pixel column, so a wide window gets the
  // detail it can show and a narrow one doesn't pay for detail it can't.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null || buffer === null) return;
    const ctx = canvas.getContext("2d")!;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const { width } = canvas.getBoundingClientRect();
      canvas.width = width * dpr;
      canvas.height = HEIGHT * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, HEIGHT);

      const bins = Math.max(1, Math.floor(width));
      const env = peaks(buffer, bins);
      const mid = HEIGHT / 2;
      const toX = (t: number) => (t / buffer.duration) * width;
      const fromX = region === null ? 0 : toX(region.start);
      const toEdge = region === null ? width : toX(region.end);

      if (region !== null) {
        ctx.fillStyle = COLORS.selectedBg;
        ctx.fillRect(fromX, 0, toEdge - fromX, HEIGHT);
      }

      for (let x = 0; x < bins; x++) {
        // A bar shorter than a pixel still reads as "there is audio here".
        const h = Math.max(1, env[x] * (HEIGHT / 2 - 4));
        const inside = region === null || (x >= fromX && x <= toEdge);
        ctx.fillStyle = inside ? COLORS.waveSelected : COLORS.wave;
        ctx.fillRect(x, mid - h, 1, h * 2);
      }

      if (region !== null) {
        ctx.fillStyle = COLORS.edge;
        ctx.fillRect(fromX, 0, 1, HEIGHT);
        ctx.fillRect(toEdge - 1, 0, 1, HEIGHT);
      }
    };

    draw();
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [buffer, region]);

  // Drag anywhere on the waveform to select; a click (or a drag too short to
  // mean anything) clears the selection back to the whole file. Pointer capture
  // keeps the drag alive when the cursor leaves the canvas.
  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (buffer === null) return;
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const secondsAt = (clientX: number) => {
      const frac = (clientX - rect.left) / rect.width;
      return Math.min(buffer.duration, Math.max(0, frac * buffer.duration));
    };
    const anchor = secondsAt(e.clientX);
    canvas.setPointerCapture(e.pointerId);

    const onMove = (ev: PointerEvent) => {
      const at = secondsAt(ev.clientX);
      const [start, end] = at < anchor ? [at, anchor] : [anchor, at];
      setRegion(end - start < MIN_SELECTION_SECONDS ? null : { start, end });
    };
    const onUp = () => {
      canvas.releasePointerCapture(e.pointerId);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
  }

  if (failed) return null;

  const selected = region ?? { start: 0, end: buffer?.duration ?? 0 };
  return (
    <section className="card flex flex-col gap-2 p-4">
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        style={{ height: HEIGHT }}
        className="w-full cursor-text touch-none select-none rounded-lg bg-surface-2"
        aria-label="Audio waveform: drag to select the part to transcribe"
      />
      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted">
        <span>
          {buffer === null ? (
            "Reading the audio…"
          ) : region === null ? (
            <>Whole track, {formatTime(buffer.duration)}. Drag to transcribe only a part.</>
          ) : (
            <>
              <span className="text-content">
                {formatTime(selected.start)} – {formatTime(selected.end)}
              </span>{" "}
              of {formatTime(buffer.duration)}
            </>
          )}
        </span>
        {region !== null && (
          <Button
            kind="ghost"
            className="px-1 py-0.5 text-sm text-muted underline underline-offset-4 hover:bg-transparent enabled:hover:text-content"
            onClick={() => setRegion(null)}
          >
            Select the whole track
          </Button>
        )}
      </div>
    </section>
  );
}
