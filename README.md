<p align="center">
  <img src="web/logo_muscriptor_final.png" alt="MuScriptor logo" width="300">
</p>

# MuScriptor

MuScriptor is a multi-instrument music transcription (audio-to-MIDI) model developed by [Kyutai](https://kyutai.org) and [Mirelo](https://www.mirelo.ai).
It's the most accurate open-source transcription model.
You can use the model [here](https://muscriptor.kyutai.org) or self-host it using this repository.


[Use it](https://muscriptor.kyutai.org) | [Paper](https://arxiv.org/abs/2607.08168v1) | [HuggingFace](https://huggingface.co/MuScriptor)

<!-- TODO: record the demo GIF (web UI piano roll), save it as assets/demo.gif,
     then uncomment:
<p align="center">
  <img src="assets/demo.gif" alt="MuScriptor web UI: live piano roll while transcribing" width="700">
</p>
-->

## HuggingFace login (required)

To use MuScriptor locally, you first need to log into [HuggingFace](https://huggingface.co/MuScriptor)
and accept the CC BY-NC 4.0 license.

1. Accept the model license on the model page for the [small](https://huggingface.co/MuScriptor/muscriptor-small),
   [medium](https://huggingface.co/MuScriptor/muscriptor-medium) or [large](https://huggingface.co/MuScriptor/muscriptor-large) model
   (access is granted automatically).
2. Authenticate on your machine:

   ```bash
   uvx hf auth login
   ```

   or set a token (create one at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

   ```bash
   export HF_TOKEN=hf_...
   ```

The weights are then automatically downloaded on first use and cached locally.

## Try it locally

After Hugging Face authentication, you can use MuScriptor with `uvx` without having to clone this repo.

Some platforms need an extra `uvx` flag, on every `uvx muscriptor` command:

| Platform | Command |
|---|---|
| Linux, macOS with Apple Silicon | `uvx muscriptor serve` |
| Windows (to use the GPU) | `uvx --torch-backend=cu128 muscriptor serve` |
| macOS with Intel | `uvx --python 3.12 muscriptor serve` |

On Windows the default PyTorch backend is `cpu`, so the GPU needs
`--torch-backend=cu128`. On Intel Macs, PyTorch stopped shipping x86_64 wheels
after torch 2.2.2, which supports Python ≤ 3.12, so the Python version has to
be pinned (if you install with pip/uv instead, use Python 3.10–3.12).

### Web UI

You can host the web UI locally with:

```bash
uvx muscriptor serve
```

This gives you the same UI as hosted on https://muscriptor.kyutai.org/, just with a different look.

### Command-line interface (CLI)

```bash
uvx muscriptor transcribe path/to/audio_file.wav
```

See `--help` for all the options.

### From Python

with uv (recommended):

```bash
uv add muscriptor
```

```bash
pip install muscriptor
```

## Models

Three variants are published under the [MuScriptor](https://huggingface.co/MuScriptor)
HuggingFace organization. Everywhere a model is selected (`load_model()`, the
CLI's `--model`, `serve --model`) you can pass the bare size keyword and the
weights are downloaded and cached automatically. The architecture is a transformer decoder only. Here are the detailed model sizes:

| Variant | Parameters | Layers | Dim | HuggingFace repo |
|---|---|---|---|---|
| `small` | 103M | 14 | 768 | [muscriptor-small](https://huggingface.co/MuScriptor/muscriptor-small) |
| `medium` (default) | 307M | 24 | 1024 | [muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium) |
| `large` | 1.4B | 48 | 1536 | [muscriptor-large](https://huggingface.co/MuScriptor/muscriptor-large) |

`small` is the practical choice on CPU-only machines, `medium` is the default
speed/accuracy trade-off, and `large` is the most accurate but really wants a
GPU. On Apple Silicon the model runs on Metal (MPS) automatically, in float16
by default (`--dtype float32` to override) — fast enough that `large`
transcribes several times faster than real time. 


## Developing

### One-time setup

```bash
uv sync
cd web && pnpm install && pnpm run build && cd ..
```

`pnpm run build` is required once — it outputs to `muscriptor/web_dist/`,
which the FastAPI server auto-mounts if it exists (and which ships inside
the PyPI wheel, so `uvx muscriptor serve` works without a checkout).

The soundfonts are not bundled: the server fetches
`MuseScore_General.sf2` (215 MB, used by `/auralize`) and
`MuseScore_General.sf3` (38 MB, the compressed build the UI plays) from
[MuScriptor/assets](https://huggingface.co/MuScriptor/assets) on first use
and caches them locally (see `muscriptor/soundfonts.py`).

### Run

```bash
uv run muscriptor serve \
    --model medium \
    --device cuda \
    --host 0.0.0.0 \
    --port 8222
```

`--model` accepts a size keyword (`small`, `medium`, `large`) that downloads
the matching variant from HuggingFace (cached under `~/.cache/muscriptor/`),
a local safetensors path, or an `hf://` / `http(s)://` URL. It defaults to
`medium` when omitted.

Then open <http://127.0.0.1:8222/> (or the LAN address of the host) and drop
a WAV onto the page.

- Drop `--device cuda` if running CPU-only.
- `--host 0.0.0.0` makes it reachable on the LAN; the default `127.0.0.1`
  is local-only.
- Playback runs a full SoundFont synthesizer ([SpessaSynth](https://github.com/spessasus/spessasynth_lib))
  in the browser, fed with `MuseScore_General.sf3` — the same soundfont the
  `/auralize` endpoint uses, served by the app itself from `/soundfonts/`
  (cached server-side), no third-party CDN.

## License

The code in this repository is released under the [MIT license](LICENSE).

The model weights, published on
[HuggingFace](https://huggingface.co/MuScriptor), are released under the
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use).

The MuseScore General SoundFont downloaded for playback / auralization is
distributed under its own (MIT) license.

## Citation

```bibtex
@misc{rouard2026muscriptoropenmodelmultiinstrument,
      title={MuScriptor: An Open Model for Multi-Instrument Music Transcription}, 
      author={Simon Rouard and Michael Krause and Axel Roebel and Carl-Johann Simon-Gabriel and Alexandre Défossez},
      year={2026},
      eprint={2607.08168},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2607.08168}, 
}
```
