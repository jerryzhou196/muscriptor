// Round-trip check for the WAV writer: `node --test src/waveform.test.ts`
// (run by `pnpm test`). It is the one piece of the waveform editor the server
// has to be able to read back, so it is the one piece worth pinning down.
import { test } from "node:test";
import assert from "node:assert/strict";
import { encodeWav, formatTime } from "./waveform.ts";

test("encodeWav writes a readable 16-bit PCM header", async () => {
  const left = new Float32Array([0, 0.5, -0.5]);
  const right = new Float32Array([1, -1, 0]);
  const view = new DataView(await encodeWav([left, right], 48000).arrayBuffer());
  const ascii = (at: number, n: number) =>
    String.fromCharCode(...new Uint8Array(view.buffer, at, n));

  assert.equal(ascii(0, 4), "RIFF");
  assert.equal(ascii(8, 4), "WAVE");
  assert.equal(view.getUint16(20, true), 1, "uncompressed PCM");
  assert.equal(view.getUint16(22, true), 2, "channels");
  assert.equal(view.getUint32(24, true), 48000, "sample rate");
  assert.equal(view.getUint16(34, true), 16, "bits per sample");
  assert.equal(view.getUint32(40, true), 3 * 2 * 2, "data size");
  assert.equal(view.byteLength, 44 + 3 * 2 * 2);
  // Interleaved L,R,L,R,…
  assert.deepEqual(
    [0, 1, 2, 3, 4, 5].map((i) => view.getInt16(44 + i * 2, true)),
    [0, 32767, 16383, -32768, -16384, 0],
  );
});

test("encodeWav clamps samples that overshoot ±1", async () => {
  const view = new DataView(
    await encodeWav([new Float32Array([2, -2])], 44100).arrayBuffer(),
  );
  assert.equal(view.getInt16(44, true), 32767);
  assert.equal(view.getInt16(46, true), -32768);
});

test("formatTime pads seconds", () => {
  assert.equal(formatTime(0), "0:00");
  assert.equal(formatTime(9.4), "0:09");
  assert.equal(formatTime(125), "2:05");
});
