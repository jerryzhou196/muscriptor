"""Tests for muscriptor/utils/midi.py."""

import numpy as np
import pytest
from mido import MidiFile

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.beats import BAR_OFFSET_MARKER, BeatGrid, read_bar_offset
from muscriptor.utils.chords import CHORD_MARKER, Chord
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


def _first_note_time(midi):
    played = 0.0
    for msg in midi:
        played += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            return played
    raise AssertionError("no note in the MIDI")


def test_late_notes_are_moved_onto_a_detected_grid():
    """The notes follow the tracked beats, which locate them better than the model.

    Real onsets land a few milliseconds after those beats, so they are moved back
    onto them. The grid itself is written as detected: `bar_offset` stays a pure
    bar-alignment shift, which is what /auralize undoes to line the synthesis up
    with the original audio.
    """
    delay = 0.012
    beats = 0.31 + np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.31, beats=beats)
    midi = notes_to_midi(_late_notes(beats, delay), beat_grid=grid)

    assert read_bar_offset(midi) == round(grid.bar_offset(), 4)
    # First onset was 0.31 + delay; corrected to 0.31, the shifted downbeat lands
    # it on a bar line rather than `delay` after one.
    played = _first_note_time(midi)
    assert played == pytest.approx(grid.bar_offset() + 0.31, abs=0.002)
    assert played % grid.bar_seconds == pytest.approx(0.0, abs=0.002)


@pytest.mark.parametrize("beats_per_bar", [4, None])
def test_correction_buys_headroom_rather_than_squashing_the_start(beats_per_bar):
    """The bar-alignment shift grows by a whole bar (or beat) to make room.

    A grid whose downbeat already sits on a bar line has nothing to absorb the
    correction, and the first note is exactly `delay` in.
    """
    delay = 0.012
    beats = np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(
        bpm=120.0, beats_per_bar=beats_per_bar, first_downbeat=0.0, beats=beats
    )
    step = grid.bar_seconds or 60.0 / grid.bpm
    midi = notes_to_midi(_late_notes(beats, delay), beat_grid=grid)

    assert read_bar_offset(midi) == pytest.approx(step, abs=0.001)
    # The first onset (delay) is corrected to 0, then shifted a whole step in, so
    # it still lands on a bar line — with room to spare instead of a clamp.
    assert _first_note_time(midi) == pytest.approx(step, abs=0.002)


def test_notes_that_ignore_the_beat_are_left_alone():
    """Nothing to measure a correction from means the notes are written as-is."""
    beats = np.arange(64) * (60.0 / 120.0)
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.31, beats=beats)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    assert read_bar_offset(midi) == round(grid.bar_offset(), 4)
    # The earliest onset is 0.0, so it plays at exactly the bar-alignment shift.
    assert _first_note_time(midi) == pytest.approx(grid.bar_offset(), abs=0.002)


def test_bar_offset_marker_is_machine_readable():
    grid = BeatGrid(bpm=120.0, beats_per_bar=3, first_downbeat=0.4)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid)
    text = _metas(midi, "marker")[0].text
    assert text.startswith(BAR_OFFSET_MARKER)
    assert read_bar_offset(midi) > 0


# --- the chord track --------------------------------------------------------


def _chord_ticks(midi):
    """`(absolute tick, symbol)` for every chord marker in `midi`."""
    found = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "marker" and msg.text.startswith(CHORD_MARKER):
                found.append((tick, msg.text.removeprefix(CHORD_MARKER)))
    return found


def test_chords_are_written_as_markers():
    chords = [
        Chord(start=0.0, end=1.0, root=0, quality="maj"),
        Chord(start=1.0, end=2.0, root=9, quality="min7"),
    ]
    grid = BeatGrid(bpm=120, beats_per_bar=None, first_downbeat=0.0, onset_delay=0.0)
    midi = notes_to_midi(_sample_notes(), beat_grid=grid, chords=chords)
    # 120 BPM, 480 ticks per beat: one second is two beats.
    assert _chord_ticks(midi) == [(0, "C"), (960, "Am7")]


def test_no_chords_writes_no_markers():
    assert _chord_ticks(notes_to_midi(_sample_notes())) == []


def test_chords_move_onto_the_beat_grid_with_the_notes():
    """A chord written over a note has to land on that note's tick, not near it.

    The notes are pulled back by `onset_delay` and pushed forward by the bar
    offset; a chord that missed either shift would sit a fraction of a beat off
    the harmony it names.
    """
    onset = 4.0
    notes = [Note(is_drum=False, program=0, onset=onset, offset=onset + 0.5, pitch=60)]
    grid = BeatGrid(
        bpm=120,
        beats_per_bar=4,
        first_downbeat=0.3,
        beats=0.3 + np.arange(32) * 0.5,
        onset_delay=0.02,
    )
    chords = [Chord(start=onset, end=onset + 1.0, root=7, quality="maj")]
    midi = notes_to_midi(notes, beat_grid=grid, chords=chords)

    note_on = next(
        (tick, m)
        for track in midi.tracks
        for tick, m in [
            (sum(x.time for x in track[: i + 1]), track[i]) for i in range(len(track))
        ]
        if m.type == "note_on" and m.velocity > 0
    )
    assert _chord_ticks(midi) == [(note_on[0], "G")]
