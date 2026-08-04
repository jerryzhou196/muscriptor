"""MIDI output utilities."""

import dataclasses

from muscriptor.tokenizer.notes import Note, note2note_event, note_event2midi
from muscriptor.utils.beats import BeatGrid

# Written when no grid was detected: 120 BPM and no time signature, leaving the
# meter for notation software to guess.
PLACEHOLDER_GRID = BeatGrid(bpm=120, beats_per_bar=None, first_downbeat=0.0)


def shifted_notes(notes: list[Note], delay_s: float) -> list[Note]:
    """`notes` moved by `delay_s`. May go negative; the bar offset covers that."""
    if not delay_s:
        return notes
    return [
        dataclasses.replace(
            note, onset=note.onset + delay_s, offset=note.offset + delay_s
        )
        for note in notes
    ]


def notes_to_midi(
    notes: list[Note],
    velocity: int = 100,
    program_names: dict[int, str] | None = None,
    beat_grid: BeatGrid | None = None,
):
    """Convert a list of Note objects to a mido MidiFile.

    `program_names` maps program numbers to human-readable track names
    (see note_event2midi).

    `grid` is a detected beat grid (see muscriptor.utils.beats): it supplies the
    tempo, the time signature and a delay that puts bar lines on real downbeats.
    Defaults to PLACEHOLDER_GRID.

    The notes are moved onto the beat grid based on its `onset_delay` to better match
    the grid.
    """
    beat_grid = beat_grid or PLACEHOLDER_GRID
    if beat_grid.onset_delay is None:
        beat_grid = beat_grid.with_onset_delay([n.onset for n in notes])
    delay = beat_grid.onset_delay
    return note_event2midi(
        note2note_event(shifted_notes(notes, -delay)),
        output_file=None,
        velocity=velocity,
        tempo=round(60_000_000 / beat_grid.bpm),
        program_names=program_names,
        beats_per_bar=beat_grid.beats_per_bar,
        offset_s=beat_grid.bar_offset(min_shift=delay),
    )
