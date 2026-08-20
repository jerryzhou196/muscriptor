/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** GA4 measurement ID; unset = analytics disabled (see src/analytics.ts). */
  readonly VITE_GA_MEASUREMENT_ID?: string;
  /** Origin of the muscriptor server (`/transcribe`, `/instruments`,
   *  `/auralize`, `/sheets`, `/soundfonts`, `/health`), e.g.
   *  `https://gpu.example.com`. Unset = same origin, which is what the
   *  server-hosted bundle and the `vite dev` proxy both want. See src/api.ts. */
  readonly VITE_TRANSCRIBE_API_BASE?: string;
  /** Origin of the standalone chord service (`/analyze`, `/health`), e.g.
   *  `https://owner-muscriptor-chords.hf.space`. Unset = no separate service,
   *  and the chords the muscriptor server sends along with the transcription
   *  are used instead. See src/api.ts. */
  readonly VITE_CHORD_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
