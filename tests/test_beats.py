"""Tests for muscriptor/utils/beats.py.

All synthetic: the maths is exercised without beat_this or a checkpoint, since
only detect_grid touches the model.
"""

import numpy as np

from muscriptor.utils.beats import (
    BAR_OFFSET_MARKER,
    MAX_TEMPO_RESIDUAL,
    BeatGrid,
    fit_tempo,
    infer_beats_per_bar,
    read_bar_offset,
)


def _beats(bpm=120.0, n=64, start=0.0, drift=0.0):
    """Beat times at `bpm`, optionally with a linear tempo ramp of `drift`."""
    t = start + np.arange(n) * (60.0 / bpm)
    if drift:
        span = t[-1] - t[0]
        t = t[0] + (t - t[0]) * (1 + drift * (t - t[0]) / span)
    return t


def test_fit_tempo_recovers_tempo():
    bpm, residual = fit_tempo(_beats(103.437))
    assert abs(bpm - 103.437) / 103.437 < 1e-3
    assert residual < 1e-6


def test_tempo_residual_gate_rejects_drifting_beats():
    """A 15% tempo ramp must fail the constant-tempo gate; steady beats pass."""
    for drift, should_pass in ((0.0, True), (0.15, False)):
        bpm, residual = fit_tempo(_beats(96.0, drift=drift))
        passes = residual < MAX_TEMPO_RESIDUAL * (60.0 / bpm)
        assert passes is should_pass


def test_infer_beats_per_bar_unanimous():
    beats = _beats(103.437, n=200)
    downbeats = beats[::4]
    assert infer_beats_per_bar(beats, downbeats) == 4


def test_infer_beats_per_bar_rejects_inconsistent_bars():
    """The Tears In The Typing Pool case: 2 beats/bar in only ~64% of bars."""
    beat = 0.68
    spacings = [2 * beat] * 50 + [3 * beat] * 24 + [4 * beat] * 5
    downbeats = np.concatenate([[0.0], np.cumsum(spacings)])
    beats = np.arange(0, downbeats[-1] + beat, beat)
    assert infer_beats_per_bar(beats, downbeats) is None


def test_bar_offset_puts_a_bar_line_on_the_first_downbeat():
    grid = BeatGrid(bpm=103.437, beats_per_bar=4, first_downbeat=1.560)
    offset = grid.bar_offset()
    bar = grid.bar_seconds
    assert 0.0 <= offset < bar
    # After the shift the first downbeat sits on a bar boundary.
    shifted = grid.first_downbeat + offset
    assert abs(shifted % bar) < 1e-9 or abs(shifted % bar - bar) < 1e-9


def test_bar_offset_is_forward_only_for_any_downbeat():
    grid = BeatGrid(bpm=90.0, beats_per_bar=3, first_downbeat=0.0)
    for first in np.linspace(0.0, 10.0, 51):
        grid.first_downbeat = float(first)
        assert 0.0 <= grid.bar_offset() < grid.bar_seconds


def test_bar_offset_is_zero_without_a_meter():
    grid = BeatGrid(bpm=120.0, beats_per_bar=None, first_downbeat=3.7)
    assert grid.bar_seconds is None
    assert grid.bar_offset() == 0.0


class _FakeMidi:
    def __init__(self, texts):
        self.tracks = [[_FakeMarker(t) for t in texts]]


class _FakeMarker:
    type = "marker"

    def __init__(self, text):
        self.text = text


def test_read_bar_offset():
    assert read_bar_offset(_FakeMidi([f"{BAR_OFFSET_MARKER}0.7945"])) == 0.7945
    assert read_bar_offset(_FakeMidi([])) == 0.0
    assert read_bar_offset(_FakeMidi(["some other marker"])) == 0.0
    assert read_bar_offset(_FakeMidi([f"{BAR_OFFSET_MARKER}nonsense"])) == 0.0
