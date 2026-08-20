"""Chord recognition from the audio, the way a DAW's chord track does it.

The chords are heard, not inferred from the transcribed notes: the audio goes
through BTC, the Bi-directional Transformer for Chord Recognition (Park et al.,
ISMIR 2019), which reads a constant-Q spectrogram and labels every ~93 ms frame
with one of 170 chords — 12 roots x 14 qualities, plus "no chord". The network
lives in `muscriptor.models.btc`; this module owns everything around it: the
features it expects, its published weights, and turning its frame labels into
chord spans that line up with the beat grid.

That last step is what makes the chords usable as notation. The model's
boundaries land wherever the audio changes, a few tens of milliseconds either
side of the beat; `snap_to_beats` moves each one onto the nearest tracked beat,
so a chord change coincides with a bar or beat line in the score instead of
landing a 32nd note off it.
"""

import dataclasses
import hashlib
import logging
from collections.abc import Sequence

import numpy as np
import torch
from mido import MidiFile, MidiTrack

from muscriptor.models.btc import TIMESTEP, BTCModel
from muscriptor.utils.audio import resample
from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.download import download_if_necessary

logger = logging.getLogger(__name__)

# Feature settings BTC was trained with (its run_config.yaml). All of them are
# constraints, not preferences: the model reads a 144-bin constant-Q transform
# at 24 bins per octave from C1 up, and nothing else.
SAMPLE_RATE = 22050
N_BINS = 144
BINS_PER_OCTAVE = 24
HOP_LENGTH = 2048

# The reference implementation computes the CQT in 10-second blocks and
# concatenates them, which is worth reproducing exactly: a 10-second block at
# this hop yields exactly TIMESTEP frames, so a block is also one model window,
# and the model's attention masks are sized for that.
WINDOW_SECONDS = 10.0
FRAME_SECONDS = HOP_LENGTH / SAMPLE_RATE

# Model windows per forward pass. Windows are independent — the attention never
# crosses one — so this only trades memory for speed.
BATCH_WINDOWS = 16

# The published "large vocabulary" checkpoint, pinned to the commit it was
# vetted at. It is a pickle rather than a safetensors file, so it is only ever
# loaded after its digest matches: see `_checkpoint_path`.
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/"
    "2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model_large_voca.pt"
)
CHECKPOINT_SHA256 = "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"

# Chord roots as the model numbers them, and as they are spelled in a score.
# Which spelling a song gets is decided once, by `prefers_flats`.
ROOTS_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
ROOTS_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# The 14 qualities of the large vocabulary, in the checkpoint's own order: a
# label index is `root * 14 + quality`. The suffix is what gets written above
# the staff (and into the MIDI markers, which are parsed back out again — so
# the suffixes have to stay distinct from one another).
QUALITY_SUFFIXES = {
    "min": "m",
    "maj": "",
    "dim": "dim",
    "aug": "aug",
    "min6": "m6",
    "maj6": "6",
    "min7": "m7",
    "minmaj7": "mMaj7",
    "maj7": "maj7",
    "7": "7",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "sus2": "sus2",
    "sus4": "sus4",
}
QUALITIES = tuple(QUALITY_SUFFIXES)

# The notes each quality is made of, as semitones above the root. What the
# symbol means, in other words — needed by anything that has to sound a chord
# rather than print it (the web player's chord track).
QUALITY_INTERVALS = {
    "min": (0, 3, 7),
    "maj": (0, 4, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "min6": (0, 3, 7, 9),
    "maj6": (0, 4, 7, 9),
    "min7": (0, 3, 7, 10),
    "minmaj7": (0, 3, 7, 11),
    "maj7": (0, 4, 7, 11),
    "7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}

# Everything above ends the vocabulary: 168 is "some chord, but not one of the
# above", 169 is "no chord at all". Both are written as a rest in the harmony.
UNKNOWN_INDEX = len(QUALITIES) * 12
NO_CHORD_INDEX = UNKNOWN_INDEX + 1
NO_CHORD_LABEL = "N.C."

# Prefix of the MIDI marker each chord change is written as, so the chord track
# travels with the notes: `muscriptor:chord=Bbm7`. Written by
# `muscriptor.tokenizer.notes.note_event2midi`, read back by
# `read_chord_markers` — which is how a MIDI upload to /sheets can still be
# engraved with its chords on it.
CHORD_MARKER = "muscriptor:chord="

# Where each pitch class sits on the line of fifths, spelled with sharps and
# spelled with flats (…F=-1, C=0, G=1, D=2…). The naturals coincide; the five
# black keys are what the two orderings disagree about, and how far they pull
# the rest of the song's roots apart is what `prefers_flats` measures.
_FIFTHS_SHARP = (0, 7, 2, 9, 4, -1, 6, 1, 8, 3, 10, 5)
_FIFTHS_FLAT = (0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5)


class ChordDetectionError(RuntimeError):
    """Chord recognition could not run (missing dependency, bad weights, …)."""


@dataclasses.dataclass(frozen=True)
class Chord:
    """One chord, sounding over `[start, end)` seconds of the original audio.

    `root` is a pitch class (0 = C) and `quality` one of QUALITIES, except for
    a silent or unrecognizable span, which has neither and reads as "N.C.".
    """

    start: float
    end: float
    root: int | None
    quality: str | None

    @property
    def is_no_chord(self) -> bool:
        return self.root is None

    @property
    def intervals(self) -> tuple[int, ...]:
        """Semitones above the root that sound in this chord; empty for N.C."""
        return QUALITY_INTERVALS[self.quality] if self.quality else ()

    def label(self, flats: bool = False) -> str:
        """The chord symbol, e.g. "Bbm7" — what a reader sees above the staff."""
        if self.root is None or self.quality is None:
            return NO_CHORD_LABEL
        roots = ROOTS_FLAT if flats else ROOTS_SHARP
        return roots[self.root] + QUALITY_SUFFIXES[self.quality]

    def step_and_alter(self, flats: bool = False) -> tuple[str, int]:
        """The root as a note letter and an alteration in semitones.

        MusicXML spells a harmony root that way (`<root-step>`, `<root-alter>`)
        rather than as a symbol, so this is the same choice `label` makes,
        taken apart.
        """
        if self.root is None:
            raise ValueError("a no-chord span has no root")
        name = (ROOTS_FLAT if flats else ROOTS_SHARP)[self.root]
        return name[0], {"": 0, "#": 1, "b": -1}[name[1:]]


def parse_label(label: str) -> tuple[int | None, str | None]:
    """Inverse of `Chord.label`: a symbol back to (root pitch class, quality).

    Returns (None, None) for "N.C.". Raises ValueError on anything this module
    would not have written, so a malformed marker fails loudly instead of
    silently engraving the wrong chord.
    """
    if label == NO_CHORD_LABEL:
        return None, None
    for length in (2, 1):  # "Bb" before "B"
        head, tail = label[:length], label[length:]
        for roots in (ROOTS_SHARP, ROOTS_FLAT):
            if head not in roots:
                continue
            for quality, suffix in QUALITY_SUFFIXES.items():
                if suffix == tail:
                    return roots.index(head), quality
    raise ValueError(f"not a chord symbol: {label!r}")


def prefers_flats(chords: Sequence[Chord]) -> bool:
    """Whether this progression reads better with flats than with sharps.

    A key's chords sit close together on the line of fifths, so the spelling
    that packs the song's roots into the tighter cluster is the one a musician
    would write: the roots of a song in B major spread over five fifths as B,
    F#, C#, G#, and over eleven as B, Gb, Db, Ab. Decided once for the whole
    song, weighted by how long each root sounds, so a score never mixes D# and
    Eb. Ties go to sharps.
    """
    weight = np.zeros(12)
    for chord in chords:
        if chord.root is not None:
            weight[chord.root] += max(0.0, chord.end - chord.start)
    if not weight.any():
        return False

    def spread(positions: tuple[int, ...]) -> float:
        """Weighted variance of the roots' positions under one spelling."""
        places = np.asarray(positions, dtype=float)
        centre = float((weight * places).sum() / weight.sum())
        return float((weight * (places - centre) ** 2).sum() / weight.sum())

    return spread(_FIFTHS_FLAT) < spread(_FIFTHS_SHARP)


def _mono_audio(wav: torch.Tensor, sample_rate: int) -> np.ndarray:
    """`wav` as a 1-D float32 array at SAMPLE_RATE, mono."""
    audio = wav.detach().cpu().float()
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate, SAMPLE_RATE)
    return audio.numpy()


def cqt_features(wav: torch.Tensor, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Log-magnitude CQT of `wav`, and the time each frame sits at.

    Returns (features [n_frames, N_BINS], times [n_frames] in seconds of the
    original audio). The transform is taken over WINDOW_SECONDS blocks and the
    results concatenated, as in the reference implementation — each block
    starts its own frame clock, which is why the times are built here alongside
    the features rather than derived from the frame index afterwards.
    """
    try:
        import librosa
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ChordDetectionError(
            "chord recognition needs librosa, which is missing from this "
            "install: pip install librosa"
        ) from e

    audio = _mono_audio(wav, sample_rate)
    block_samples = int(SAMPLE_RATE * WINDOW_SECONDS)
    blocks, times = [], []
    for start in range(0, max(len(audio), 1), block_samples):
        chunk = audio[start : start + block_samples]
        # librosa needs a couple of hops to place a frame at all; a shorter
        # tail carries no chord anyone would notate, so it is dropped.
        if len(chunk) < 2 * HOP_LENGTH:
            break
        spectrum = librosa.cqt(
            chunk,
            sr=SAMPLE_RATE,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
            hop_length=HOP_LENGTH,
        )
        blocks.append(spectrum)
        times.append(start / SAMPLE_RATE + np.arange(spectrum.shape[1]) * FRAME_SECONDS)
    if not blocks:
        return np.zeros((0, N_BINS), dtype=np.float32), np.zeros(0)
    feature = np.log(np.abs(np.concatenate(blocks, axis=1)) + 1e-6)
    return feature.T.astype(np.float32), np.concatenate(times)


def _checkpoint_path():
    """The published weights, downloaded once and checked against their digest.

    The reference distributes a pickle, which `torch.load` executes on load, so
    the digest is verified first: this refuses to run anything but the exact
    file the URL was pinned to.
    """
    path = download_if_necessary(CHECKPOINT_URL)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ChordDetectionError(
            f"the chord model downloaded from {CHECKPOINT_URL} has digest "
            f"{digest}, expected {CHECKPOINT_SHA256}. Delete {path} and retry."
        )
    return path


_MODEL_CACHE: dict[str, tuple[BTCModel, float, float]] = {}


def load_model(device: str | torch.device = "cpu") -> tuple[BTCModel, float, float]:
    """The BTC network plus the (mean, std) its features are normalized with.

    Cached per device: the weights are 12 MB and loading them takes long enough
    to be worth not repeating for every request the server handles.
    """
    key = str(device)
    if key not in _MODEL_CACHE:
        checkpoint = torch.load(
            _checkpoint_path(), map_location="cpu", weights_only=False
        )
        model = BTCModel()
        model.load_published_state_dict(checkpoint["model"])
        model.eval().to(device)
        _MODEL_CACHE[key] = (model, float(checkpoint["mean"]), float(checkpoint["std"]))
    return _MODEL_CACHE[key]


def _predict(features: np.ndarray, device: str | torch.device) -> np.ndarray:
    """Chord index per frame, from normalized features [n_frames, N_BINS]."""
    model, mean, std = load_model(device)
    normalized = (features - mean) / std
    # The model reads fixed-size windows, so the tail is padded out to one and
    # its filler frames dropped again afterwards.
    padding = -len(normalized) % TIMESTEP
    if padding:
        normalized = np.pad(normalized, ((0, padding), (0, 0)))
    windows = torch.from_numpy(normalized).reshape(-1, TIMESTEP, N_BINS)

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(windows), BATCH_WINDOWS):
            batch = windows[start : start + BATCH_WINDOWS].to(device)
            predictions.append(model(batch).argmax(dim=-1).flatten().cpu())
    labels = torch.cat(predictions).numpy()
    return labels[: len(features)]


def _to_spans(labels: np.ndarray, times: np.ndarray, duration: float) -> list[Chord]:
    """Frame labels to chord spans, merging every run of equal labels."""
    chords = []
    starts = np.flatnonzero(np.diff(labels, prepend=labels[0] - 1))
    for i, first in enumerate(starts):
        index = int(labels[first])
        end = float(times[starts[i + 1]]) if i + 1 < len(starts) else duration
        if index >= UNKNOWN_INDEX:
            root, quality = None, None
        else:
            root, quality = divmod(index, len(QUALITIES))
            quality = QUALITIES[quality]
        chords.append(Chord(float(times[first]), end, root, quality))
    return chords


def merge_adjacent(chords: Sequence[Chord]) -> list[Chord]:
    """Fuse neighbouring spans that name the same chord, and drop empty ones."""
    merged: list[Chord] = []
    for chord in chords:
        if chord.end <= chord.start:
            continue
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.root == chord.root
            and previous.quality == chord.quality
        ):
            merged[-1] = dataclasses.replace(previous, end=chord.end)
        else:
            merged.append(chord)
    return merged


def snap_to_beats(chords: Sequence[Chord], grid: BeatGrid) -> list[Chord]:
    """Move every chord boundary onto the nearest beat of `grid`.

    A chord change that the model put 40 ms before the beat belongs *on* the
    beat once it is written down; without this the engraver would have to
    render it as a tied 32nd note. Spans that collapse (both ends snapping to
    the same beat) are dropped, which is also what removes the flicker the
    model sometimes produces at a transition.
    """
    beats = grid.beats
    if beats is None or len(beats) < 2:
        # A grid built by hand carries no beat times; extrapolate them from its
        # tempo instead, which is all the notation is quantized to anyway.
        span = 60.0 / grid.bpm
        last = max((chord.end for chord in chords), default=grid.first_downbeat)
        count = max(2, int((last - grid.first_downbeat) / span) + 2)
        beats = grid.first_downbeat + np.arange(count) * span
    beats = np.asarray(beats, dtype=float)

    def snap(time: float) -> float:
        return float(beats[int(np.abs(beats - time).argmin())])

    snapped = [
        dataclasses.replace(chord, start=snap(chord.start), end=snap(chord.end))
        for chord in chords
    ]
    return merge_adjacent(snapped)


def detect_chords(
    wav: torch.Tensor,
    sample_rate: int,
    grid: BeatGrid | None = None,
    device: str | torch.device = "cpu",
) -> list[Chord]:
    """Recognize the chords in `wav`, aligned to `grid` when there is one.

    Returns spans covering the audio in order, including the "N.C." stretches
    where nothing harmonic is playing. Without a beat grid the boundaries stay
    where the model heard them, which is fine for a chord track but too ragged
    to notate — `snap_to_beats` is what makes them line up with the bar lines.
    """
    features, times = cqt_features(wav, sample_rate)
    if not len(features):
        return []
    duration = wav.shape[-1] / sample_rate
    chords = merge_adjacent(_to_spans(_predict(features, device), times, duration))
    if grid is not None:
        chords = snap_to_beats(chords, grid)
    named = sum(1 for chord in chords if not chord.is_no_chord)
    logger.info("recognized %d chords (%d distinct spans)", named, len(chords))
    return chords


def published_chords(chords: Sequence[Chord]) -> list[Chord]:
    """`chords` from the first real one on.

    The silence a recording opens with is not a chord change, so leading "N.C."
    spans are dropped; the ones between chords stay, since they say the harmony
    stopped. Everything that hands the chord track to someone else — the MIDI
    markers, the server's event stream — starts here, so they all agree on
    where the chords begin.
    """
    for index, chord in enumerate(chords):
        if not chord.is_no_chord:
            return list(chords[index:])
    return []


def to_markers(
    chords: Sequence[Chord], flats: bool | None = None
) -> list[tuple[float, str]]:
    """`(seconds, chord symbol)` for every chord change, for the MIDI markers.

    `flats` overrides the spelling that `prefers_flats` would pick.
    """
    if flats is None:
        flats = prefers_flats(chords)
    return [(chord.start, chord.label(flats)) for chord in published_chords(chords)]


def read_chord_markers(midi: MidiFile) -> list[tuple[float, str]]:
    """`(position in quarter notes, chord symbol)` for the markers in `midi`.

    Quarter notes rather than seconds: this reads a written score's chord
    track, where the position that matters is the musical one — and the MIDI
    ticks already are that, tempo changes and bar offset included.
    """
    markers = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "marker" and message.text.startswith(CHORD_MARKER):
                symbol = message.text.removeprefix(CHORD_MARKER)
                markers.append((tick / midi.ticks_per_beat, symbol))
    markers.sort(key=lambda marker: marker[0])
    return markers


def strip_chord_markers(midi: MidiFile) -> MidiFile:
    """`midi` without its chord markers, leaving everything else where it was.

    Notation software renders markers as text in the score, so the copy handed
    to the engraver has them taken out — the chords go back in as proper
    harmony elements instead (see `muscriptor.utils.harmony`).
    """
    stripped = MidiFile(ticks_per_beat=midi.ticks_per_beat, type=midi.type)
    for track in midi.tracks:
        kept = MidiTrack()
        # A dropped message's delta has to be handed to the message after it,
        # or everything downstream of it slides earlier.
        carried = 0
        for message in track:
            if message.type == "marker" and message.text.startswith(CHORD_MARKER):
                carried += message.time
                continue
            kept.append(message.copy(time=message.time + carried))
            carried = 0
        stripped.tracks.append(kept)
    return stripped
