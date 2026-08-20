/** A failed `/transcribe` request. `userMessage`, when set, is a server-sent
 *  explanation safe to show the user (e.g. "could not decode audio file …"). */
export class TranscribeError extends Error {
  readonly userMessage?: string;
  readonly status?: number;
  constructor(message: string, opts: { userMessage?: string; status?: number } = {}) {
    super(message);
    this.name = "TranscribeError";
    this.userMessage = opts.userMessage;
    this.status = opts.status;
  }
}

export interface TranscribeOptions {
  /** Extra multipart form fields to send alongside the file (e.g. instruments). */
  extra?: Record<string, string | string[]>;
  signal?: AbortSignal;
  /** Sent as the `X-Client-Id` header so a resubmit can supersede this tab's
   *  older active or queued request without affecting other users. */
  clientId?: string;
  /** Called once the queued job reaches the front and starts transcribing. */
  onAccepted?: () => void;
  /** Called when the server reports how many requests are ahead in its FIFO. */
  onQueued?: (info: { position: number }) => void;
  /** Called on each 503 refusal (another transcription is in progress), with the
   *  wait until the next attempt. Only `streamTranscribeWithRetry` reports it. */
  onBusy?: (info: { attempt: number; retryInMs: number }) => void;
}

/** Stream SSE `data:` JSON payloads from a POST upload of `file`. */
export async function* streamTranscribe(
  url: string,
  file: File,
  opts: TranscribeOptions = {},
): AsyncGenerator<unknown> {
  const { extra, signal, clientId, onAccepted, onQueued } = opts;
  const form = new FormData();
  form.append("file", file, file.name);
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      if (Array.isArray(v)) {
        for (const item of v) form.append(k, item);
      } else {
        form.append(k, v);
      }
    }
  }
  const headers: Record<string, string> = {};
  if (clientId) headers["X-Client-Id"] = clientId;
  const resp = await fetch(url, { method: "POST", body: form, signal, headers });
  if (!resp.ok || !resp.body) {
    // FastAPI's HTTPException bodies are `{"detail": "..."}`. Pull the detail
    // out (when present) so the UI can show why the upload was rejected
    // instead of a bare status code.
    let detail: string | undefined;
    try {
      const body = await resp.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON body (or network error reading it) — leave detail unset.
    }
    throw new TranscribeError(`server returned ${resp.status}`, {
      userMessage: detail,
      status: resp.status,
    });
  }
  let accepted = false;
  const accept = () => {
    if (accepted) return;
    accepted = true;
    onAccepted?.();
  };
  const reader = resp.body
    .pipeThrough(new TextDecoderStream())
    .getReader();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let sep: number;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const chunk = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          const message = JSON.parse(line.slice(6));
          if (
            message?.type === "queued" &&
            Number.isInteger(message.position) &&
            message.position > 0
          ) {
            onQueued?.({ position: message.position });
            continue;
          }
          if (message?.type === "transcription_started") {
            accept();
            continue;
          }
          // Compatibility with servers from before the queue protocol: their
          // first real transcription event means the model has started.
          accept();
          yield message;
        }
      }
    }
  }
}

/** Resolves after `ms`, or rejects immediately (or as soon as it fires) with
 *  an `AbortError` if `signal` is aborted — so a retry wait can't outlive the
 *  request it's retrying. */
function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

const RETRY_INTERVAL_MS = 5000;

/** Like `streamTranscribe`, but transparently retries every
 *  `RETRY_INTERVAL_MS` when an older server reports 503 instead of queueing.
 *
 *  This keeps the frontend usable during a rolling backend deployment. */
export async function* streamTranscribeWithRetry(
  url: string,
  file: File,
  opts: TranscribeOptions = {},
): AsyncGenerator<unknown> {
  for (let attempt = 1; ; attempt++) {
    try {
      yield* streamTranscribe(url, file, opts);
      return;
    } catch (e) {
      if (!(e instanceof TranscribeError) || e.status !== 503) throw e;
      opts.onBusy?.({ attempt, retryInMs: RETRY_INTERVAL_MS });
      await delay(RETRY_INTERVAL_MS, opts.signal);
    }
  }
}
