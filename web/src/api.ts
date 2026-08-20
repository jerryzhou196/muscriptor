import type { ChordChangeAudio } from "./audio";
import type { ChordChange } from "./pianoroll";

/**
 * Where the backends live.
 *
 * The UI has two of them, and which origin each sits on is a deployment
 * decision, not a code one:
 *
 *   - the muscriptor server (`/transcribe`, `/instruments`, `/auralize`,
 *     `/sheets`, `/soundfonts`, `/health`) — a GPU box, because note
 *     transcription needs one;
 *   - the chord service (`/analyze`, `/health`) — CPU-only, so it can live on
 *     a free Hugging Face Space instead of renting GPU time to run BTC.
 *
 * Both default to `""`, which makes every URL below a same-origin relative
 * path — exactly what the code used to hardcode. That keeps the two existing
 * deployments working untouched: FastAPI serving the built bundle from
 * `muscriptor/web_dist` via StaticFiles, and `vite dev`, whose proxy forwards
 * those same paths to a backend. Only a standalone static deployment (Vercel)
 * sets them, and then to absolute origins — which is also the moment CORS
 * starts to matter, since the browser is now talking cross-origin.
 */
function normalizeBase(raw: string | undefined): string {
  // A trailing slash would double up against the leading slash of every path,
  // and `//transcribe` is a different (broken) URL — so drop it here rather
  // than relying on whoever fills in the Vercel dashboard field to.
  return (raw ?? "").trim().replace(/\/+$/, "");
}

const TRANSCRIBE_BASE = normalizeBase(import.meta.env.VITE_TRANSCRIBE_API_BASE);
const CHORD_BASE = normalizeBase(import.meta.env.VITE_CHORD_API_BASE);

/** Absolute URL of a muscriptor-server path (`path` starts with a slash). */
export function transcribeApi(path: string): string {
  return TRANSCRIBE_BASE + path;
}

/** Absolute URL of a chord-service path (`path` starts with a slash). */
export function chordApi(path: string): string {
  return CHORD_BASE + path;
}

/**
 * True when a standalone chord service is configured. When it is, chords come
 * from there; when it isn't, they come from the ones the muscriptor server
 * already embeds in its `transcription_complete` event. It is one or the
 * other — never both, so there is only ever one chord track to reason about.
 */
export const chordServiceEnabled = CHORD_BASE !== "";

/**
 * One recognized chord change, in the single shape the app consumes: the piano
 * roll draws `label` at `time`, the audio engine voices `root` + `intervals`.
 * The chord service returns exactly the fields the SSE event does (plus a
 * `duration` the UI doesn't need — it derives the end from the next change),
 * so both sources land on this one type.
 */
export type RecognizedChord = ChordChange & ChordChangeAudio;

/** FastAPI's `{"detail": "..."}`, when the failing response carries one. */
async function errorDetail(resp: Response): Promise<string> {
  try {
    const body = await resp.json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    // Not JSON, or the body was already consumed — fall through.
  }
  return `HTTP ${resp.status}`;
}

/** How many times to poke a cold chord service before giving up on warming it. */
const WARM_ATTEMPTS = 3;
/** Gap between those pokes. A Space boots in tens of seconds, so this is coarse. */
const WARM_RETRY_MS = 8000;

/**
 * Ping the chord service's `/health` so it is awake by the time a file is
 * uploaded.
 *
 * A free Hugging Face Space is stopped when idle and takes tens of seconds to
 * come back — the first request pays for the boot. Sending that first request
 * while the user is still choosing a file (and then again while it reports
 * `{"status": "loading"}`, which the service answers with before its model has
 * finished loading) usually hides the whole wake-up.
 *
 * Resolves `true` once the service reports ready, `false` if it never did.
 * It never rejects and never surfaces anything to the user: chords are an
 * optional extra, and a sleeping chord service must not look like a broken app.
 */
export async function warmChordService(signal?: AbortSignal): Promise<boolean> {
  if (!chordServiceEnabled) return false;
  for (let attempt = 1; attempt <= WARM_ATTEMPTS; attempt++) {
    if (signal?.aborted) return false;
    try {
      const resp = await fetch(chordApi("/health"), { signal });
      if (resp.ok) {
        const body = (await resp.json()) as { status?: string };
        // "loading" = awake, model still coming up. Worth another ping; a
        // request sent now would just queue behind the load anyway.
        if (body.status !== "loading") return true;
      }
    } catch {
      // Asleep, booting, or unreachable — all indistinguishable from here, and
      // all handled the same way: try again, then let it go.
    }
    if (attempt < WARM_ATTEMPTS) await sleep(WARM_RETRY_MS, signal);
  }
  return false;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });
}

/**
 * Recognize the chords in `file` with the standalone chord service.
 *
 * `detect_tempo` is left at the service's `best-effort` default: when it finds
 * a beat grid the chord boundaries are snapped to it, and the returned times
 * are already grid-aligned — so, like the chords that come over SSE, they need
 * no `onset_delay` correction before being drawn or played.
 *
 * Throws on a refusal (413 too long, 429 rate limited, 503 all slots busy,
 * 504 timeout, or a cold Space that never answered); the caller decides how
 * loudly to fail, and for chords the answer is "not at all".
 */
export async function analyzeChords(
  file: File,
  signal?: AbortSignal,
): Promise<RecognizedChord[]> {
  const form = new FormData();
  form.append("file", file, file.name);
  const resp = await fetch(chordApi("/analyze"), {
    method: "POST",
    body: form,
    signal,
  });
  if (!resp.ok) throw new Error(await errorDetail(resp));
  const body = (await resp.json()) as { chords?: unknown };
  const chords = Array.isArray(body.chords) ? body.chords : [];
  // Take only the fields the UI uses, and coerce them: this response crosses an
  // origin boundary from a separately deployed service, so it is the one place
  // where a version skew could hand us something unexpected.
  return chords.map((raw) => {
    const c = raw as Partial<RecognizedChord>;
    return {
      time: Number(c.time) || 0,
      label: String(c.label ?? ""),
      root: typeof c.root === "number" ? c.root : null,
      intervals: Array.isArray(c.intervals) ? c.intervals.map(Number) : [],
    };
  });
}
