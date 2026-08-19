import { useEffect, useRef, type Ref } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import ZoomPlugin from "wavesurfer.js/dist/plugins/zoom.esm.js";
import { encodeWav, peakEnvelope } from "../waveform";

const WAVE = "#5a5c68";
const PROGRESS = "#ff5b7a";
const REGION = "rgba(255, 91, 122, 0.16)";
const HEIGHT = 96;

function regionsOf(wave: WaveSurfer) {
  return wave
    .getActivePlugins()
    .find((plugin): plugin is RegionsPlugin => plugin instanceof RegionsPlugin);
}

type WaveformParams = {
  file: File;
  buffer: AudioBuffer | null;
  /** True once the buffer is no longer the decoded upload. */
  edited: boolean;
  range: [number, number] | null;
  onRangeChange: (range: [number, number] | null) => void;
  onPlayingChange: (playing: boolean) => void;
  onError: () => void;
  /** False when the editor has unmounted the container (decode / load failure). */
  active: boolean;
};

/**
 * WaveSurfer bound to a container: created when the node is on screen, torn
 * down when it is not. Playback does not follow the editor's buffer identity.
 */
export function useWaveform({
  file,
  buffer,
  edited,
  range,
  onRangeChange,
  onPlayingChange,
  onError,
  active,
}: WaveformParams) {
  const containerRef = useRef<HTMLDivElement>(null);
  const waveRef = useRef<WaveSurfer | null>(null);

  useEffect(() => {
    if (!active || containerRef.current === null) return;

    const wave = WaveSurfer.create({
      container: containerRef.current,
      waveColor: WAVE,
      progressColor: PROGRESS,
      cursorColor: PROGRESS,
      height: HEIGHT,
      normalize: true,
    });
    const regions = wave.registerPlugin(RegionsPlugin.create());
    // Scroll or pinch to zoom at the pointer. Wheel events whose horizontal
    // delta dominates are left alone, so a sideways swipe and the scrollbar
    // pan instead.
    wave.registerPlugin(
      ZoomPlugin.create({ exponentialZooming: true, iterations: 24 }),
    );
    waveRef.current = wave;
    regions.enableDragSelection({ color: REGION });

    // RegionsPlugin scrolls the waveform to follow a region as it moves. Keep
    // the viewport fixed during that gesture; normal scrollbar and trackpad
    // panning still work when the user is not dragging a region.
    let regionDragScroll: number | null = null;
    const wrapper = wave.getWrapper();
    const rememberRegionDragScroll = (event: PointerEvent) => {
      const path = event.composedPath();
      regionDragScroll = regions
        .getRegions()
        .some((region) => region.element !== null && path.includes(region.element))
        ? wave.getScroll()
        : null;
    };
    wrapper.addEventListener("pointerdown", rememberRegionDragScroll);

    wave.on("error", () => onError());
    wave.on("play", () => onPlayingChange(true));
    wave.on("pause", () => onPlayingChange(false));
    wave.on("finish", () => onPlayingChange(false));

    regions.on("region-created", (r) => {
      for (const other of [...regions.getRegions()]) {
        if (other !== r) other.remove();
      }
      onRangeChange([r.start, r.end]);
    });
    regions.on("region-update", (_r, side) => {
      // The plugin has already adjusted the scrollbar by this point. Restore
      // it synchronously, before the browser paints the intermediate frame.
      if (side === undefined && regionDragScroll !== null) {
        wave.setScroll(regionDragScroll);
      }
    });
    regions.on("region-updated", (r) => {
      regionDragScroll = null;
      onRangeChange([r.start, r.end]);
    });
    regions.on("region-removed", () => {
      if (regions.getRegions().length === 0) onRangeChange(null);
    });

    return () => {
      wrapper.removeEventListener("pointerdown", rememberRegionDragScroll);
      waveRef.current = null;
      wave.destroy();
    };
  }, [active]);

  useEffect(() => {
    const wave = waveRef.current;
    if (wave === null || buffer === null) return;

    regionsOf(wave)?.clearRegions();

    let cancelled = false;
    const blob = edited
      ? encodeWav(
          Array.from({ length: buffer.numberOfChannels }, (_, c) =>
            buffer.getChannelData(c),
          ),
          buffer.sampleRate,
        )
      : file;
    wave.loadBlob(blob, peakEnvelope(buffer), buffer.duration).catch(() => {
      if (!cancelled) onError();
    });

    return () => {
      cancelled = true;
    };
  }, [buffer, edited, file]);

  // Parent cleared the selection (button, cut, crop); drop the plugin box.
  useEffect(() => {
    if (range !== null) return;
    const wave = waveRef.current;
    if (wave !== null) regionsOf(wave)?.clearRegions();
  }, [range]);

  function playPause() {
    const wave = waveRef.current;
    if (wave === null) return;
    const region = regionsOf(wave)?.getRegions()[0];
    if (wave.isPlaying()) wave.pause();
    else if (region !== undefined) region.play(true);
    else wave.play();
  }

  function pause() {
    waveRef.current?.pause();
  }

  return { containerRef, playPause, pause };
}

/** Empty node WaveSurfer paints into. */
export function Waveform({ ref }: { ref?: Ref<HTMLDivElement> }) {
  return (
    <div
      ref={ref}
      className="w-full [scrollbar-color:#5a5c68_transparent]"
      style={{ minHeight: HEIGHT }}
    />
  );
}
