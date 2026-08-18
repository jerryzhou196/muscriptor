/**
 * Decoding and WAV encoding for the waveform editor.
 *
 * The editor trims in the browser rather than sending offsets to the server:
 * the file is already decoded here to feed WaveSurfer its peaks, so cutting the
 * region out costs one array copy, and the upload shrinks to the part the user
 * actually wants transcribed. `/transcribe` reads PCM WAV through the stdlib
 * `wave` module (see `muscriptor/utils/audio.py`), which is what `encodeWav`
 * writes.
 */

/**
 * Decode an audio file to raw samples.
 *
 * Uses an OfflineAudioContext because the welcome screen has no user gesture
 * yet — a real AudioContext would be created suspended and count against
 * Safari's per-page limit, while decoding never touches the output device.
 */
export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
  const ctx = new OfflineAudioContext(1, 1, 44100);
  return await ctx.decodeAudioData(await file.arrayBuffer());
}

/**
 * `buffer` with the `[start, end)` seconds removed and the two remaining
 * pieces butted together.
 *
 * Cutting on a sample boundary can click if the join lands mid-cycle; that is
 * audible in the preview but harmless to transcribe, which is what this audio
 * is for. Callers must not cut the whole buffer away — a zero-length
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
 * The name keeps the original stem so the downloaded MIDI is still recognisable
 * (`midiFilenameRef` derives from it), with the trimmed span appended.
 */
export function trimToWavFile(
  file: File,
  buffer: AudioBuffer,
  start: number,
  end: number,
): File {
  const from = Math.max(0, Math.floor(start * buffer.sampleRate));
  const to = Math.min(buffer.length, Math.ceil(end * buffer.sampleRate));
  const channels = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    channels.push(buffer.getChannelData(c).slice(from, to));
  }
  const blob = encodeWav(channels, buffer.sampleRate);
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
