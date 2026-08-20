# MuScriptor web UI

React + Vite frontend for MuScriptor. `pnpm run build` writes the bundle
straight into `../muscriptor/web_dist`, which the Python package ships in its
wheel (`[tool.hatch.build] artifacts` in `pyproject.toml`) and serves with
FastAPI's `StaticFiles` — so the default build is the one the
`uvx muscriptor serve` deployment needs, and nothing here changes that.

```bash
pnpm install
pnpm dev        # dev server on :5173, proxying the API paths to :8222
pnpm run build  # type-check, then build into ../muscriptor/web_dist
```

## Where the backends are

Every request the UI makes goes through `src/api.ts`, which prefixes the path
with a base URL fixed at build time:

| Variable | Backend | Paths |
| --- | --- | --- |
| `VITE_TRANSCRIBE_API_BASE` | muscriptor server (GPU) | `/transcribe`, `/instruments`, `/auralize`, `/sheets`, `/soundfonts`, `/health` |
| `VITE_CHORD_API_BASE` | chord service (CPU, HF Space) | `/analyze`, `/health` |

Both default to `""`, i.e. same-origin relative paths — exactly what the code
did before they existed. That is what the FastAPI-hosted bundle and the
`vite dev` proxy (see `vite.config.ts`) both need, so leaving them unset keeps
those deployments byte-for-byte as they were. See `.env.example`.

`VITE_CHORD_API_BASE` also decides *where chords come from*, and it is one or
the other, never both:

- **unset** — the chords embedded in the muscriptor server's
  `transcription_complete` SSE event are used, as before;
- **set** — those are ignored and the audio is analyzed by the chord service
  instead, in parallel with the (much slower) note transcription.

## Deploying to Vercel

The site lives in this subdirectory, so **set the Vercel project's Root
Directory to `web`** and keep `vercel.json` here rather than at the repo root.
That is the choice that makes the tooling line up:

- Vercel picks the package manager from the lockfile and `packageManager`
  field it finds in the root directory — from `web/` it gets pnpm 10 and
  `pnpm install --frozen-lockfile` for free; from the repo root (which has no
  `package.json`) it would fall back to npm and need an install-command
  override.
- `outputDirectory` is resolved inside the root directory, so it cannot point
  at `../muscriptor/web_dist`. `vercel.json` therefore overrides `outDir` in
  its build command only (`vite build --outDir dist`), leaving the default
  `pnpm run build` — and with it the Python wheel packaging that depends on
  `muscriptor/web_dist` — untouched.

Then set both variables under Settings → Environment Variables (for Production
*and* Preview — Vite inlines them at build time, so a change needs a redeploy,
not just a save).

There is deliberately **no SPA rewrite**: this app is a single page with no
client-side router, so rewriting every path to `index.html` would only turn
genuine 404s (a mistyped asset URL, say) into a 200 that renders the app.
Hashed bundles under `/assets/` are served `immutable` for a year;
`index.html` must revalidate so a deploy is picked up immediately.

### CORS, and the preview-deployment gotcha

On Vercel the frontend is on a different origin from both backends, so both
have to allow it explicitly:

- the muscriptor server must allow the Vercel domain, including the
  `X-Client-Id` request header that `/transcribe` sends (it shows up in the
  preflight);
- the chord service uses a strict CSV allowlist (`ALLOWED_ORIGINS`, never
  `*`), so the Vercel domain has to be in it.

**The gotcha:** every preview deployment gets its own generated origin
(`muscriptor-web-<hash>-<scope>.vercel.app`), and a fresh hash is not in
anyone's allowlist. Previews will therefore fail CORS against both backends
until the origin is added. Options, in rough order of preference:

1. assign a stable domain to the preview branch (Vercel: Settings → Domains →
   add a domain pointed at a branch) and allowlist that one name once;
2. add the specific preview URL to `ALLOWED_ORIGINS` while it is being
   reviewed, and take it out afterwards;
3. accept the degradation for previews — a blocked chord service is handled
   gracefully (see below), but a blocked transcription server is not: the
   preview will show the "server temporarily unavailable" notice.

### Cold starts

A free Hugging Face Space is stopped when idle and takes tens of seconds to
boot. The welcome screen pings `GET {chord base}/health` on mount to start
that boot while the user is still choosing a file, retrying a couple of times
while the service answers `{"status": "loading"}` (awake, model still loading).

Nothing about this can fail loudly: the warm-up result never reaches the UI,
and if the analysis itself fails — asleep, rate limited, all slots busy,
CORS-blocked — the transcription still completes normally and simply has no
chords, leaving the (opt-in) chord toggle greyed out. Chords are applied
whenever they arrive, which may be a moment after the notes.
