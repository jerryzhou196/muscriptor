"""Tests for muscriptor/utils/midi.py."""

import tempfile
from pathlib import Path

import pytest
from mido import MidiFile

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.beats import BAR_OFFSET_MARKER, BeatGrid, read_bar_offset
from muscriptor.utils.midi import notes_to_midi, save_midi


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


def test_every_note_track_repeats_the_tempo():
    """MuseScore ignores set_tempo in a note-less conductor track."""
    grid = BeatGrid(bpm=90, beats_per_bar=None, first_downbeat=0.0)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    for track in midi.tracks[1:]:
        tempos = [m for m in track if m.type == "set_tempo"]
        assert [m.tempo for m in tempos] == [round(60_000_000 / 90)]


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


def test_bar_offset_marker_is_machine_readable():
    grid = BeatGrid(bpm=120.0, beats_per_bar=3, first_downbeat=0.4)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    text = _metas(midi, "marker")[0].text
    assert text.startswith(BAR_OFFSET_MARKER)
    assert read_bar_offset(midi) > 0


def test_save_midi_creates_file():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.mid"
        save_midi(notes, path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_save_midi_is_valid_midi():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.mid"
        save_midi(notes, path)
        loaded = MidiFile(str(path))
        assert len(loaded.tracks) > 0


def test_save_midi_string_path():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "out.mid")
        save_midi(notes, path)
        assert Path(path).exists()
