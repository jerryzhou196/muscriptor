"""Beat-grid detection, for writing real tempo and time signatures into MIDI."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Used to detect songs that don't have a constant tempo (don't use a metronome)
MAX_TEMPO_RESIDUAL = 0.05

# Fraction of bars that must agree on a beats-per-bar count to write a time
# signature. Trackers that lose the meter spread their downbeats across several
# spacings, and a wrong time signature is worse than none.
MIN_METER_AGREEMENT = 0.9

MIN_BEATS = 8

# Marker text prefix recording how far notes were delayed to align bar lines,
# so `/auralize` can line the synthesis back up with the original audio.
BAR_OFFSET_MARKER = "muscriptor:bar_offset="

# ------------------------------------------------- onset-phase downbeat correction
#
# Transcribed onsets land a few milliseconds after the beats the tracker reports
# on the same recording — some of it the model's own delay, some the tracker
# placing beats differently. Bar lines put on the tracked downbeat therefore sit
# slightly *before* the notes they are meant to bracket.
# `BeatGrid.aligned_to_onsets` measures that gap from the notes themselves and
# moves the grid onto them.
#
# The measurement is the onset-phase route of the offline delay analysis, where it
# goes by "predicted onsets vs recording grid": express every onset as a position
# within its beat, treat each as a unit vector at angle 2π·frac(position × s) on a
# grid of `s` subdivisions per beat, and average. The resultant's length says how
# tightly the onsets sit on that grid and its angle says how late they sit. It is
# the one estimate of the delay that needs nothing but a transcription and the
# beats already detected for the tempo — no annotations, no resynthesis.

# Candidate grids, binary and triplet divisions of the beat, simplest first.
ONSET_SUBDIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16, 24)

# Accept a simpler grid whose concentration is within this much of the best one:
# |R| climbs again on every grid that merely contains a coarser one, and the
# coarser one is the honest description of where the onsets are.
CONCENTRATION_SLACK = 0.95

# The angle only pins the offset down modulo one subdivision, so half a
# subdivision is the largest delay it can express; grids finer than this could
# not hold the delay unambiguously and are not considered.
MIN_HALF_SUBDIVISION_S = 0.03

# Below this resultant length the onsets are not really on the chosen grid, and
# its angle means nothing.
MIN_ONSET_CONCENTRATION = 0.25

# |R| for n random angles is about 1/sqrt(n), so fewer onset times than this can
# clear the floor above by chance. Roughly a bar's worth of eighth notes per
# second of audio, i.e. a few seconds of dense music or a good deal more of
# sparse music.
MIN_ONSETS = 100

# Offsets larger than this are refused as a bad grid fit rather than a real
# delay. Over the 353 songs of the sweep where this is readable at all the
# measurement runs +6.7 ms at the median, quartiles +0.7 and +12.3 ms, 99th
# percentile +25 ms, so nothing past 40 ms is the delay this is meant to correct;
# what does produce such numbers is a mis-chosen coarse grid, whose angle can
# come out anywhere.
MAX_ONSET_OFFSET_S = 0.04

# Note onset times in seconds, however the caller happens to hold them.
Onsets = Sequence[float] | np.ndarray


class BeatDetectionError(RuntimeError):
    """No usable beat grid in the audio (too short, or no constant tempo)."""


# What to do when the tempo can't be detected: True raises, False doesn't even
# try (the escape hatch for songs the detector gets wrong), and "best-effort"
# warns and falls back to the placeholder tempo.
TempoDetection = bool | Literal["best-effort"]


def read_bar_offset(midi) -> float:
    """Seconds of bar-alignment delay recorded in a MidiFile, 0.0 if absent."""
    for track in midi.tracks:
        for msg in track:
            if msg.type == "marker" and msg.text.startswith(BAR_OFFSET_MARKER):
                try:
                    return float(msg.text.removeprefix(BAR_OFFSET_MARKER))
                except ValueError:
                    return 0.0
    return 0.0


@dataclass
class BeatGrid:
    """A constant-tempo beat grid detected from audio."""

    bpm: float
    # None when the meter could not be determined; write no time signature.
    beats_per_bar: int | None
    # Time of the first detected bar line, in seconds.
    first_downbeat: float
    # The individual beat times the grid was fitted to, kept for
    # `aligned_to_onsets`, which needs them rather than the fitted tempo: the
    # tracked beats follow the recording's small tempo wobbles, and those are the
    # same size as the offset being measured. None for a hand-built grid; kept
    # out of equality and repr, being an array and an implementation detail.
    beats: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def bar_seconds(self) -> float | None:
        if self.beats_per_bar is None:
            return None
        return self.beats_per_bar * 60.0 / self.bpm

    def bar_offset(self) -> float:
        """Seconds to delay every note so bar lines land on downbeats.

        MIDI has no pickup measure: bar 1 starts at tick 0, so the only way to
        put a bar line on the first downbeat is to shift the music later. Always
        a forward shift, keeping ticks non-negative and dropping no notes; the
        leading partial bar holds whatever preceded the first downbeat.
        """
        bar = self.bar_seconds
        if bar is None:
            return 0.0
        return (bar - self.first_downbeat % bar) % bar

    def aligned_to_onsets(self, onsets: Onsets) -> "BeatGrid":
        """This grid moved onto `onsets`, so bar lines land on the notes.

        `onsets` are note onset times in seconds on the same timeline as the
        audio the grid was detected from — the transcription the grid is about to
        be written alongside. Whatever offset separates them from the tracked
        beats is added to `first_downbeat`, which is a pure phase shift: the
        tempo and the meter are unchanged, and so is the audio the notes
        describe. What changes is `bar_offset`, by the same amount in the other
        direction, so the notes come out sitting on the bar line instead of just
        after it.

        Returns `self` unchanged when there is nothing to measure from: no
        tracked beats (a hand-built grid), too few onsets, or onsets that do not
        sit on a beat subdivision at all. See :func:`measure_onset_offset`.
        """
        if self.beats is None:
            return self
        measured = measure_onset_offset(onsets, self.beats)
        if measured is None:
            return self
        logger.info(
            "onsets sit %+.1f ± %.1f ms off a 1/%d-beat grid (|R| = %.2f over %d "
            "onsets); moving the downbeat from %.3fs to %.3fs",
            1000 * measured.offset_s,
            1000 * measured.sem_s,
            measured.subdivision,
            measured.concentration,
            measured.n_onsets,
            self.first_downbeat,
            self.first_downbeat + measured.offset_s,
        )
        return replace(self, first_downbeat=self.first_downbeat + measured.offset_s)


@dataclass(frozen=True)
class OnsetOffset:
    """How far a set of note onsets sits from the beat subdivision it is on."""

    # Signed seconds, positive when the onsets are late.
    offset_s: float
    # Standard error of `offset_s`: how precisely this transcription pins it down.
    sem_s: float
    # Resultant length in [0, 1]: how tightly the onsets sit on the grid.
    concentration: float
    # Subdivisions per beat of the grid the offset is measured against.
    subdivision: int
    # Distinct onset times that went into it.
    n_onsets: int


def onset_phase(onsets: Onsets, beats: np.ndarray) -> np.ndarray:
    """Onset times as positions in continuous beats.

    Onsets outside the tracked span drop out, since there is no beat to place
    them in. One entry per distinct onset time (to the millisecond) rather than
    per note, so a six-note chord does not outvote six single notes elsewhere.
    """
    times = np.unique(np.round(np.asarray(onsets, dtype=float), 3))
    inside = (times >= beats[0]) & (times <= beats[-1])
    return np.interp(times[inside], beats, np.arange(len(beats)))


def phase_resultant(
    phase_beats: np.ndarray, subdivision: int
) -> tuple[float, float, float]:
    """(concentration, offset_beats, sem_beats) of onset phase on a 1/s grid."""
    angles = 2 * np.pi * np.mod(phase_beats * subdivision, 1.0)
    mean = np.exp(1j * angles).mean()
    concentration, theta = float(np.abs(mean)), float(np.angle(mean))
    # Standard error of a mean direction: the spread tangential to it, thinned by
    # sqrt(n) and divided by the resultant length — a short resultant pins the
    # angle down badly.
    spread = np.sqrt((np.sin(angles - theta) ** 2).mean() / len(angles))
    turns = 1 / (2 * np.pi * subdivision)  # radians on the fine grid → beats
    return concentration, theta * turns, spread / concentration * turns


def measure_onset_offset(onsets: Onsets, beats: np.ndarray) -> OnsetOffset | None:
    """How late `onsets` sit against the beat subdivision they are on.

    The grid is chosen per transcription: the simplest of ONSET_SUBDIVISIONS
    whose concentration is within CONCENTRATION_SLACK of the best-scoring one.
    Phase is measured within the beat rather than the bar, because trackers pin
    beats down far better than bars, and the answer is the same either way — a
    constant offset of every beat is a constant offset of every bar line.

    Returns None when the onsets cannot say anything: fewer than MIN_ONSETS of
    them on the tracked span, no grid coarse enough to express the offset, a
    concentration below MIN_ONSET_CONCENTRATION (the onsets are not on a
    subdivision), or an offset past MAX_ONSET_OFFSET_S (the grid is a bad fit).
    """
    if len(beats) < 2:
        return None
    # Mean spacing, not the median inter-beat interval: on the tracker's 20 ms
    # frame grid the median snaps to whole frames.
    period_s = (beats[-1] - beats[0]) / (len(beats) - 1)

    phase_beats = onset_phase(onsets, beats)
    if len(phase_beats) < MIN_ONSETS:
        logger.info(
            "not correcting the downbeat: %d onset time(s) on the tracked span, "
            "need %d",
            len(phase_beats),
            MIN_ONSETS,
        )
        return None

    candidates = [
        s for s in ONSET_SUBDIVISIONS if period_s / (2 * s) >= MIN_HALF_SUBDIVISION_S
    ]
    if not candidates:
        # Only at an implausible tempo: even a whole beat is under 60 ms.
        logger.info("not correcting the downbeat: no grid coarse enough to fit")
        return None
    scored = {s: phase_resultant(phase_beats, s) for s in candidates}
    best = max(concentration for concentration, _, _ in scored.values())
    subdivision = next(
        s for s in candidates if scored[s][0] >= CONCENTRATION_SLACK * best
    )
    concentration, offset_beats, sem_beats = scored[subdivision]

    if concentration < MIN_ONSET_CONCENTRATION:
        logger.info(
            "not correcting the downbeat: onsets sit on no subdivision of the beat "
            "(best |R| = %.2f, need %.2f)",
            best,
            MIN_ONSET_CONCENTRATION,
        )
        return None
    offset_s = offset_beats * period_s
    if abs(offset_s) > MAX_ONSET_OFFSET_S:
        logger.warning(
            "not correcting the downbeat: onsets measured %+.0f ms off a "
            "1/%d-beat grid, further than the %+.0f ms this can plausibly be",
            1000 * offset_s,
            subdivision,
            1000 * MAX_ONSET_OFFSET_S,
        )
        return None
    return OnsetOffset(
        offset_s=offset_s,
        sem_s=sem_beats * period_s,
        concentration=concentration,
        subdivision=subdivision,
        n_onsets=len(phase_beats),
    )


def fit_tempo(beats: np.ndarray) -> tuple[float, float]:
    """Least-squares tempo over the beat sequence.

    Returns (bpm, residual RMS in seconds). Fitting a line through beat index
    against time beats taking the median inter-beat interval: trackers quantise
    beats to a frame grid (50 Hz for beat_this), which alone limits median-IBI
    tempo resolution to a few BPM.
    """
    index = np.arange(len(beats))
    slope, intercept = np.polyfit(index, beats, 1)
    residual = beats - (intercept + slope * index)
    return 60.0 / float(slope), float(residual.std())


def infer_beats_per_bar(
    beats: np.ndarray,
    downbeats: np.ndarray,
    min_agreement: float = MIN_METER_AGREEMENT,
) -> int | None:
    """Beats per bar from downbeat spacing, or None if the bars disagree.

    Only measures how far apart the downbeats are; it cannot tell whether the
    downbeats themselves are on the right beat. Note that a tracker that
    subdivides the bar wrongly (reporting two beats per bar for music in 3/4)
    can still be self-consistent here, which is why this stays conservative.
    """
    if len(downbeats) < 3 or len(beats) < 2:
        return None
    beat = float(np.median(np.diff(beats)))
    counts = np.round(np.diff(downbeats) / beat).astype(int)
    counts = counts[counts >= 2]
    if not len(counts):
        return None
    values, tally = np.unique(counts, return_counts=True)
    best = int(tally.argmax())
    if tally[best] / len(counts) < min_agreement:
        return None
    return int(values[best])


def detect_grid(
    wav: torch.Tensor, sr: int, checkpoint: str = "final0", device: str = "cpu"
) -> BeatGrid:
    """Detect a constant-tempo beat grid.

    Args:
        wav: Audio as [C, T] (this repo's convention), float32.
        sr: Sample rate of `wav`; beat_this resamples internally.
        checkpoint: beat_this checkpoint name. "final0" over "small0": the small
            model emits spurious beats before the first downbeat, which shifts
            the bar offset by a beat or two.
        device: Torch device for the beat model.

    Raises BeatDetectionError when the audio is too short or the beats do not
    fit a constant tempo. An unclear meter is not fatal: the BeatGrid comes back
    with beats_per_bar=None, since tempo alone is worth writing.
    """
    # Imported here, not at module scope: beat_this pulls in torchaudio and soxr,
    # which would slow every CLI invocation that never transcribes anything.
    from beat_this.inference import Audio2Beats

    # This triggers an error in beat_this so report as BeatDetectionError directly
    min_duration_s = 1.0
    if wav.shape[-1] < min_duration_s * sr:
        raise BeatDetectionError(
            f"Audio is {wav.shape[-1] / sr:.2f}s long, too short to detect a tempo"
        )

    signal = wav.mean(dim=0).detach().cpu().numpy()  # beat_this wants mono, 1-D
    # Returns (beats, downbeats) despite beat_this's own File2File unpacking
    # them the other way round.
    beats, downbeats = Audio2Beats(
        checkpoint_path=checkpoint, device=device, dbn=False
    )(signal, sr)

    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(beats) < MIN_BEATS:
        raise BeatDetectionError(
            f"Only {len(beats)} beats detected, need at least {MIN_BEATS}"
        )

    bpm, residual = fit_tempo(beats)
    beat_seconds = 60.0 / bpm
    if residual > MAX_TEMPO_RESIDUAL * beat_seconds:
        raise BeatDetectionError(
            f"The recording has no fixed tempo (beats deviate {residual * 1000:.0f} ms "
            f"RMS from a constant {bpm:.1f} BPM)"
        )

    beats_per_bar = infer_beats_per_bar(beats, downbeats)
    first_downbeat = float(downbeats[0]) if len(downbeats) else float(beats[0])
    logger.info(
        "detected %.3f BPM, %s, first downbeat %.3fs (beat residual %.1f ms)",
        bpm,
        f"{beats_per_bar}/4" if beats_per_bar else "meter unknown",
        first_downbeat,
        residual * 1000,
    )
    return BeatGrid(
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        first_downbeat=first_downbeat,
        beats=beats,
    )
