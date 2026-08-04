"""Tests for muscriptor/utils/midi.py."""

import numpy as np
import pytest
from mido import MidiFile

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.beats import BAR_OFFSET_MARKER, BeatGrid, read_bar_offset
from muscriptor.utils.midi import notes_to_midi


def _metas(midi, msg_type):
    return [m for track in midi.tracks for m in track if m.type == msg_type]


def _sample_notes():
    return [
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=60),
        Note(is_drum=False, program=0, onset=0.5, offset=1.0, pitch=64),
        Note(is_drum=True, program=128, onset=0.0, offset=0.01, pitch=36),
    ]


def test_notes_to_midi_returns_midi_file():
    midi = notes_to_midi(_sample_notes())
    assert isinstance(midi, MidiFile)


def test_notes_to_midi_has_tracks():
    midi = notes_to_midi(_sample_notes())
    assert len(midi.tracks) > 0


def test_notes_to_midi_custom_tempo():
    grid = BeatGrid(bpm=90, beats_per_bar=None, first_downbeat=0.0)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    assert _metas(midi, "set_tempo")[0].tempo == round(60_000_000 / 90)


def test_notes_to_midi_empty_notes():
    midi = notes_to_midi([])
    assert isinstance(midi, MidiFile)


def test_no_grid_writes_no_time_signature():
    """Default output must stay as it was: placeholder tempo, meter unstated."""
    midi = notes_to_midi(_sample_notes())
    assert _metas(midi, "time_signature") == []
    assert _metas(midi, "marker") == []
    assert _metas(midi, "set_tempo")[0].tempo == 500000


def test_grid_writes_tempo_time_signature_and_marker():
    grid = BeatGrid(bpm=103.437, beats_per_bar=4, first_downbeat=1.560)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    tempo = _metas(midi, "set_tempo")[0].tempo
    assert round(60_000_000 / tempo) == 103  # 580063 us/beat
    sig = _metas(midi, "time_signature")[0]
    assert (sig.numerator, sig.denominator) == (4, 4)
    assert read_bar_offset(midi) == round(grid.bar_offset(), 4)


def test_grid_without_meter_writes_tempo_only():
    grid = BeatGrid(bpm=98.5, beats_per_bar=None, first_downbeat=1.2)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    assert _metas(midi, "set_tempo")
    assert _metas(midi, "time_signature") == []
    assert _metas(midi, "marker") == []


def test_bar_alignment_shifts_notes_without_negative_ticks():
    grid = BeatGrid(bpm=103.437, beats_per_bar=4, first_downbeat=1.560)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)  # earliest onset is 0.0
    assert all(m.time >= 0 for track in midi.tracks for m in track)
    # The note at t=0 is delayed by exactly the recorded offset.
    played = 0.0
    for msg in midi:
        played += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            break
    assert played == pytest.approx(grid.bar_offset(), abs=0.01)


def _late_notes(beats, delay, subdivision=4):
    """Sixteenth notes on `beats`, every one `delay` seconds late."""
    fine = np.linspace(beats[0], beats[-1], (len(beats) - 1) * subdivision + 1)
    return [
        Note(
            is_drum=False,
            program=0,
            onset=float(onset + delay),
            offset=float(onset + delay + 0.05),
            pitch=60,
        )
        for onset in fine
    ]


def test_a_detected_grid_is_moved_onto_the_notes():
    """Bar lines follow the transcription, not the beats tracked on the audio.

    Real onsets land a few milliseconds after those beats, so a grid carrying
    them (only a detected one does) is shifted onto the notes before the bar
    offset is computed — the notes then sit on the bar line rather than after it.
    """
    delay = 0.012
    beats = np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.0, beats=beats)
    midi = notes_to_midi(_late_notes(beats, delay), beat_grid=grid)

    bar = grid.bar_seconds
    assert grid.bar_offset() == 0.0  # a bar line is already on the tracked beat
    assert read_bar_offset(midi) == pytest.approx(bar - delay, abs=0.002)
    played = 0.0
    for msg in midi:
        played += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            break
    assert played % bar == pytest.approx(0.0, abs=0.002)


def test_notes_that_ignore_the_beat_leave_the_grid_alone():
    """Nothing to measure a correction from means the grid is written as detected."""
    beats = np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.31, beats=beats)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    assert read_bar_offset(midi) == round(grid.bar_offset(), 4)


def test_bar_offset_marker_is_machine_readable():
    grid = BeatGrid(bpm=120.0, beats_per_bar=3, first_downbeat=0.4)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    text = _metas(midi, "marker")[0].text
    assert text.startswith(BAR_OFFSET_MARKER)
    assert read_bar_offset(midi) > 0
