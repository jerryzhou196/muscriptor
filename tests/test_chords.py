"""Tests for muscriptor/utils/chords.py.

No model and no audio: the BTC network is exercised in test_btc_model.py, and
everything here is the layer around it — labels, beat alignment, and the MIDI
markers the chord track travels in.
"""

import io

import numpy as np
import pytest
from mido import MetaMessage, MidiFile, MidiTrack

from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.chords import (
    CHORD_MARKER,
    QUALITY_INTERVALS,
    NO_CHORD_INDEX,
    NO_CHORD_LABEL,
    QUALITIES,
    UNKNOWN_INDEX,
    Chord,
    _to_spans,
    merge_adjacent,
    parse_label,
    prefers_flats,
    published_chords,
    read_chord_markers,
    snap_to_beats,
    strip_chord_markers,
    to_markers,
)


def _chord(start, end, root, quality="maj"):
    return Chord(start=start, end=end, root=root, quality=quality)


def _grid(bpm=120.0, first=0.0, count=32):
    beat = 60.0 / bpm
    return BeatGrid(
        bpm=bpm,
        beats_per_bar=4,
        first_downbeat=first,
        beats=first + np.arange(count) * beat,
    )


def test_label_spells_root_and_quality():
    assert _chord(0, 1, 10, "min7").label() == "A#m7"
    assert _chord(0, 1, 10, "min7").label(flats=True) == "Bbm7"
    assert _chord(0, 1, 0, "maj").label() == "C"
    assert _chord(0, 1, 6, "hdim7").label() == "F#m7b5"
    assert Chord(0, 1, None, None).label() == NO_CHORD_LABEL


def test_every_quality_round_trips_through_its_symbol():
    """Every label this module writes has to parse back to the same chord.

    The MIDI markers are written as symbols and read back as chords, so a
    suffix that collides with another one would silently change a chord.
    """
    for root in range(12):
        for quality in QUALITIES:
            for flats in (False, True):
                label = _chord(0, 1, root, quality).label(flats)
                assert parse_label(label) == (root, quality)
    assert parse_label(NO_CHORD_LABEL) == (None, None)


def test_parse_label_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_label("H7")
    with pytest.raises(ValueError):
        parse_label("Cmaj9")


def test_step_and_alter_splits_the_root():
    assert _chord(0, 1, 10, "maj").step_and_alter() == ("A", 1)
    assert _chord(0, 1, 10, "maj").step_and_alter(flats=True) == ("B", -1)
    assert _chord(0, 1, 4, "maj").step_and_alter() == ("E", 0)


def test_spelling_follows_the_key_not_the_loudest_chord():
    """B major spells its roots with sharps even though F# sounds longest."""
    b_major = [
        _chord(0, 8, 6),  # F#
        _chord(8, 12, 11, "maj7"),  # Bmaj7
        _chord(12, 16, 8),  # G#
        _chord(16, 20, 1, "min7"),  # C#m7
    ]
    assert prefers_flats(b_major) is False

    e_flat_major = [
        _chord(0, 8, 3),  # Eb
        _chord(8, 12, 8),  # Ab
        _chord(12, 16, 10, "7"),  # Bb7
        _chord(16, 20, 0, "min7"),  # Cm7
    ]
    assert prefers_flats(e_flat_major) is True


def test_spelling_of_nothing_is_sharps():
    assert prefers_flats([]) is False
    assert prefers_flats([Chord(0, 4, None, None)]) is False


def test_merge_adjacent_fuses_repeats_and_drops_empties():
    merged = merge_adjacent(
        [_chord(0, 1, 0), _chord(1, 2, 0), _chord(2, 2, 5), _chord(2, 3, 7)]
    )
    assert [(c.start, c.end, c.root) for c in merged] == [(0, 2, 0), (2, 3, 7)]


def test_snap_to_beats_moves_boundaries_onto_the_grid():
    grid = _grid()  # beats every 0.5 s
    snapped = snap_to_beats([_chord(0.04, 1.98, 0), _chord(1.98, 3.96, 7)], grid)
    assert [(c.start, c.end) for c in snapped] == [(0.0, 2.0), (2.0, 4.0)]


def test_snap_to_beats_drops_a_chord_too_short_to_survive():
    """A flicker between two chords collapses instead of becoming a 32nd note."""
    grid = _grid()
    snapped = snap_to_beats(
        [_chord(0.0, 1.0, 0), _chord(1.0, 1.08, 5), _chord(1.08, 2.0, 7)], grid
    )
    assert [(c.start, c.end, c.root) for c in snapped] == [(0.0, 1.0, 0), (1.0, 2.0, 7)]


def test_snap_to_beats_without_tracked_beats_uses_the_tempo():
    """A hand-built grid carries no beat times, only a tempo to quantize to."""
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.0)
    snapped = snap_to_beats([_chord(0.03, 1.46, 0)], grid)
    assert [(c.start, c.end) for c in snapped] == [(0.0, 1.5)]


def test_to_spans_reads_the_vocabulary_layout():
    """Label index is root * 14 + quality; 168 and 169 are not chords."""
    labels = np.array(
        [11 * len(QUALITIES) + QUALITIES.index("maj7"), UNKNOWN_INDEX, NO_CHORD_INDEX]
    )
    times = np.array([0.0, 1.0, 2.0])
    spans = _to_spans(labels, times, duration=3.0)
    assert [c.label() for c in spans] == ["Bmaj7", NO_CHORD_LABEL, NO_CHORD_LABEL]
    assert [(c.start, c.end) for c in spans] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


def test_every_quality_knows_what_notes_it_is_made_of():
    """The web player sounds a chord from these, so a missing one is silence."""
    assert tuple(QUALITY_INTERVALS) == QUALITIES
    for quality, intervals in QUALITY_INTERVALS.items():
        assert intervals[0] == 0, quality  # measured from the root
        assert list(intervals) == sorted(set(intervals)), quality
        assert all(0 <= i < 12 for i in intervals), quality
        assert len(intervals) in (3, 4), quality  # triads and sevenths


def test_chord_intervals_come_from_its_quality():
    assert _chord(0, 1, 0, "maj").intervals == (0, 4, 7)
    assert _chord(0, 1, 9, "min7").intervals == (0, 3, 7, 10)
    assert Chord(0, 1, None, None).intervals == ()


def test_published_chords_start_at_the_first_real_one():
    chords = [Chord(0.0, 2.0, None, None), _chord(2.0, 4.0, 0)]
    assert published_chords(chords) == chords[1:]
    assert published_chords([Chord(0.0, 2.0, None, None)]) == []


def test_to_markers_drops_the_silence_a_song_opens_with():
    chords = [
        Chord(0.0, 2.0, None, None),
        _chord(2.0, 4.0, 0),
        Chord(4.0, 5.0, None, None),
        _chord(5.0, 6.0, 7),
    ]
    assert to_markers(chords) == [(2.0, "C"), (4.0, NO_CHORD_LABEL), (5.0, "G")]


def _midi_with_markers(*markers: tuple[int, str]) -> MidiFile:
    """A MIDI file whose meta track carries `(delta ticks, text)` markers."""
    midi = MidiFile(ticks_per_beat=480, type=1)
    track = MidiTrack()
    track.append(MetaMessage("set_tempo", tempo=500000, time=0))
    for delta, text in markers:
        track.append(MetaMessage("marker", text=text, time=delta))
    midi.tracks.append(track)
    return midi


def test_read_chord_markers_returns_quarter_positions():
    midi = _midi_with_markers(
        (0, f"{CHORD_MARKER}C"), (960, f"{CHORD_MARKER}Am7"), (240, "some other marker")
    )
    assert read_chord_markers(midi) == [(0.0, "C"), (2.0, "Am7")]


def test_strip_chord_markers_leaves_the_rest_where_it_was():
    midi = _midi_with_markers(
        (0, f"{CHORD_MARKER}C"), (960, f"{CHORD_MARKER}G"), (480, "muscriptor:other")
    )
    stripped = strip_chord_markers(midi)
    kept = [m for m in stripped.tracks[0] if m.type == "marker"]
    assert [m.text for m in kept] == ["muscriptor:other"]
    # The dropped markers' deltas were handed on, so the survivor stays at 1440.
    assert sum(m.time for m in stripped.tracks[0]) == 1440


def test_strip_chord_markers_round_trips_through_a_file():
    midi = _midi_with_markers((0, f"{CHORD_MARKER}C"), (480, f"{CHORD_MARKER}F"))
    buf = io.BytesIO()
    strip_chord_markers(midi).save(file=buf)
    assert read_chord_markers(MidiFile(file=io.BytesIO(buf.getvalue()))) == []
