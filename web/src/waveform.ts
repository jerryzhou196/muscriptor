/**
 * Buffer maths for the waveform editor.
 *
 * Edits happen in the browser rather than as offsets sent to the server: the
 * file is decoded here anyway to draw it, so a cut is one array copy and the
 * upload shrinks to the part worth transcribing. `/transcribe` reads PCM WAV
 * (`muscriptor/utils/audio.py`), which is what `encodeWav` writes.
 */

/**
 * Decode an audio file to raw samples.
 *
 * OfflineAudioContext because the welcome screen has no user gesture yet: a
 * real AudioContext would be created suspended and count against Safari's
 * per-page limit, while decoding never touches the output device.
 */
export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
  const ctx = new OfflineAudioContext(1, 1, 44100);
  return await ctx.decodeAudioData(await file.arrayBuffer());
}

/** The `[start, end)` seconds of each channel, as plain sample arrays. */
function sliceChannels(buffer: AudioBuffer, start: number, end: number): Float32Array[] {
  const from = Math.max(0, Math.round(start * buffer.sampleRate));
  const to = Math.min(buffer.length, Math.round(end * buffer.sampleRate));
  return Array.from({ length: buffer.numberOfChannels }, (_, c) =>
    buffer.getChannelData(c).slice(from, to),
  );
}

/**
 * Just the `[start, end)` seconds of `buffer` — crop to the selection.
 *
 * The region must not be empty: a zero-length AudioBuffer is invalid.
 */
export function copyRegion(buffer: AudioBuffer, start: number, end: number): AudioBuffer {
  const channels = sliceChannels(buffer, start, end);
  const out = new AudioBuffer({
    length: channels[0].length,
    numberOfChannels: channels.length,
    sampleRate: buffer.sampleRate,
  });
  channels.forEach((data, c) => out.getChannelData(c).set(data));
  return out;
}

/**
 * `buffer` with the `[start, end)` seconds removed and the remaining pieces
 * butted together.
 *
 * The join can click if it lands mid-cycle: audible in the preview, harmless
 * to transcribe. The whole buffer must not be cut away — a zero-length
 * AudioBuffer is invalid.
 */
export function cutRegion(buffer: AudioBuffer, start: number, end: number): AudioBuffer {
  const rate = buffer.sampleRate;
  const from = Math.max(0, Math.round(start * rate));
  const to = Math.min(buffer.length, Math.round(end * rate));
  const out = new AudioBuffer({
    length: buffer.length - (to - from),
    numberOfChannels: buffer.numberOfChannels,
    sampleRate: rate,
  });
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const src = buffer.getChannelData(c);
    const dst = out.getChannelData(c);
    dst.set(src.subarray(0, from), 0);
    dst.set(src.subarray(to), from);
  }
  return out;
}

/**
 * `buffer`'s channels reduced to a min/max envelope for drawing.
 *
 * WaveSurfer scans whatever array it is given for each pixel column's min and
 * max, so raw samples make every zoom step rescan millions of them (~50ms per
 * wheel event on a 72s track, and a trackpad fires dozens per gesture). This
 * is the same peak cache AudioMass keeps, computed once per buffer.
 *
 * Each bucket contributes its minimum *and* maximum, in that order, so both
 * halves of the waveform survive; plain subsampling would alias into a
 * quieter, wrong-shaped signal. `pointsPerSecond` is display resolution, not
 * the audio's — the editor cannot zoom past ~1 second across the card, so 2000
 * is already several points per pixel at the deepest zoom.
 *
 * Peaks are only ever drawn. Playback and the upload read `buffer` itself.
 */
export function peakEnvelope(buffer: AudioBuffer, pointsPerSecond = 2000): Float32Array[] {
  const buckets = Math.ceil((buffer.duration * pointsPerSecond) / 2);
  const per = Math.floor(buffer.length / buckets);
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) =>
    buffer.getChannelData(c),
  );
  // Short or low-rate audio can already be coarser than the target; reducing
  // it further would throw away detail the display can show.
  if (per < 2) return channels;

  return channels.map((src) => {
    const out = new Float32Array(buckets * 2);
    for (let b = 0; b < buckets; b++) {
      const from = b * per;
      // The last bucket takes the remainder, so no samples go unlooked-at.
      const to = b === buckets - 1 ? buffer.length : from + per;
      let min = src[from];
      let max = min;
      for (let i = from + 1; i < to; i++) {
        const v = src[i];
        if (v < min) min = v;
        else if (v > max) max = v;
      }
      out[b * 2] = min;
      out[b * 2 + 1] = max;
    }
    return out;
  });
}

/** 16-bit PCM WAV bytes for interleaved-on-write `channels` of equal length. */
export function encodeWav(channels: Float32Array[], sampleRate: number): Blob {
  const numChannels = channels.length;
  const frames = channels[0].length;
  const dataBytes = frames * numChannels * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);

  const ascii = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM header size
  view.setUint16(20, 1, true); // format: uncompressed PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true); // byte rate
  view.setUint16(32, numChannels * 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < numChannels; c++) {
      // Clamp before scaling: decoded float samples can overshoot ±1 (lossy
      // formats do this routinely) and would wrap around as int16.
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buf], { type: "audio/wav" });
}

/**
 * The `[start, end)` seconds of `buffer` as a WAV file named after `file`.
 *
 * The name keeps the original stem, so the downloaded MIDI is still
 * recognisable (`midiFilenameRef` derives from it).
 */
export function trimToWavFile(
  file: File,
  buffer: AudioBuffer,
  start: number,
  end: number,
): File {
  const blob = encodeWav(sliceChannels(buffer, start, end), buffer.sampleRate);
  const stem = file.name.replace(/\.[^/.]+$/, "");
  return new File([blob], `${stem} (${formatTime(start)}-${formatTime(end)}).wav`, {
    type: "audio/wav",
  });
}

/** `m:ss` for a duration in seconds — the editor's labels and trimmed names. */
export function formatTime(seconds: number): string {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}
