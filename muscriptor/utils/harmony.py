"""Chord symbols in a MusicXML score: `<harmony>` elements over the top staff.

MusicXML keeps harmony out of the notes: a `<harmony>` element sits in the
measure next to the note it starts on, and the engraver draws it above the
staff. Everything here is about placing it at the right musical instant —
the recognizer works in seconds, the score works in divisions of a quarter
note, and the two meet at a position in quarter notes from the start of bar 1
(which is what a chord's tick in the MIDI file already is).

Only the first part gets the symbols. That is where a reader expects them, and
repeating them over every staff would be noise.
"""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from muscriptor.utils.chords import NO_CHORD_LABEL, QUALITY_SUFFIXES, parse_label

# MusicXML's own names for the qualities BTC distinguishes. The vocabulary was
# chosen to be writable: every one of them has a standard `<kind>`.
MUSICXML_KINDS = {
    "min": "minor",
    "maj": "major",
    "dim": "diminished",
    "aug": "augmented",
    "min6": "minor-sixth",
    "maj6": "major-sixth",
    "min7": "minor-seventh",
    "minmaj7": "major-minor",
    "maj7": "major-seventh",
    "7": "dominant",
    "dim7": "diminished-seventh",
    "hdim7": "half-diminished",
    "sus2": "suspended-second",
    "sus4": "suspended-fourth",
}

# Notes advance the musical clock; these two move it explicitly (a `<backup>`
# is how a second voice starts over at the beginning of the measure).
_DURATION_TAGS = ("note", "forward", "backup")


def _duration(element: ET.Element) -> int:
    """The `<duration>` an element occupies, in divisions. 0 when it has none."""
    text = element.findtext("duration")
    return int(text) if text else 0


def _anchors(measure: ET.Element) -> tuple[list[tuple[int, int]], int]:
    """Where a harmony can be attached in `measure`, and how long the measure is.

    Returns `[(child index, position in divisions)]` for every note that starts
    a new musical instant, plus the measure's total length. Scanning stops
    contributing anchors at the first `<backup>`: that is the second voice
    starting the measure over, and a chord symbol belongs with the first.
    """
    anchors: list[tuple[int, int]] = []
    position = 0
    length = 0
    first_voice = True
    for index, child in enumerate(measure):
        if child.tag == "note":
            # A `<chord/>` note stacks onto the note before it, and a grace note
            # is squeezed in before one: neither begins an instant of its own.
            if (
                first_voice
                and child.find("chord") is None
                and child.find("grace") is None
            ):
                anchors.append((index, position))
            if child.find("chord") is None:
                position += _duration(child)
        elif child.tag == "forward":
            position += _duration(child)
        elif child.tag == "backup":
            position -= _duration(child)
            first_voice = False
        length = max(length, position)
    return anchors, length


def _divisions(measure: ET.Element, current: int) -> int:
    """The divisions per quarter note in force after `measure`'s attributes."""
    for attributes in measure.findall("attributes"):
        text = attributes.findtext("divisions")
        if text:
            current = int(text)
    return current


def _harmony_element(label: str) -> ET.Element:
    """A `<harmony>` for the chord `label`, ready to be placed in a measure."""
    harmony = ET.Element("harmony", {"print-frame": "no"})
    root = ET.SubElement(harmony, "root")
    step = ET.SubElement(root, "root-step")
    if label == NO_CHORD_LABEL:
        # "No chord" has no root to print, but MusicXML still wants the element:
        # an empty `text` attribute keeps the letter off the page.
        step.text = "C"
        step.set("text", "")
        ET.SubElement(harmony, "kind", {"text": NO_CHORD_LABEL}).text = "none"
        return harmony

    _, quality = parse_label(label)  # also rejects anything malformed
    # The label already carries the spelling chosen for the whole song, so the
    # letter and the accidental are read straight back out of it.
    step.text = label[0]
    accidental = {"#": 1, "b": -1}.get(label[1:2])
    if accidental is not None:
        ET.SubElement(root, "root-alter").text = str(accidental)
    ET.SubElement(
        harmony, "kind", {"text": QUALITY_SUFFIXES[quality]}
    ).text = MUSICXML_KINDS[quality]
    return harmony


def add_chord_symbols(path: Path, symbols: Sequence[tuple[float, str]]) -> int:
    """Write chord symbols into the MusicXML score at `path`, in place.

    `symbols` is `(position in quarter notes from the start of the score, chord
    label)`, in order. Returns how many were placed; a symbol past the end of
    the score is dropped rather than extending it.

    A symbol that falls between two notes is attached to the note before it and
    given an `<offset>`, which is how MusicXML expresses a chord change that
    doesn't coincide with an attack — mid-bar changes over a held note, above
    all.
    """
    original = path.read_text()
    tree = ET.parse(path)
    part = tree.getroot().find("part")
    if part is None:
        return 0

    # Where each measure begins, in quarter notes, and what a division is worth
    # inside it — both accumulated by walking the part, so a pickup bar or a
    # change of time signature needs no special case.
    measures = []
    start = 0.0
    divisions = 1
    for measure in part.findall("measure"):
        divisions = _divisions(measure, divisions)
        anchors, length = _anchors(measure)
        measures.append((measure, start, divisions, anchors))
        start += length / divisions

    placed = 0
    # Last symbol first, so that inserting one into a measure cannot shift the
    # child indices of an anchor this loop has yet to use. (Sorted rather than
    # merely reversed, so that holds however the caller ordered them.)
    for quarters, label in sorted(symbols, key=lambda symbol: symbol[0], reverse=True):
        if quarters >= start - 1e-6:
            continue  # past the last bar line: there is nothing to write it over
        # The measure the chord falls in: the last one that begins before it.
        found = None
        for entry in measures:
            if entry[1] > quarters + 1e-6:
                break
            found = entry
        if found is None:
            continue
        measure, measure_start, measure_divisions, anchors = found
        if not anchors:
            continue  # an empty measure has no note to write the chord over
        offset = round((quarters - measure_start) * measure_divisions)
        # The note this chord is written over: the last one that has already
        # started by the time the chord changes.
        index, position = anchors[0]
        for candidate_index, candidate_position in anchors:
            if candidate_position > offset:
                break
            index, position = candidate_index, candidate_position
        harmony = _harmony_element(label)
        if offset != position:
            ET.SubElement(harmony, "offset").text = str(offset - position)
        measure.insert(index, harmony)
        placed += 1

    # ElementTree drops the XML declaration and the DOCTYPE that MusicXML
    # readers expect, so the original prologue is put back verbatim.
    body = ET.tostring(tree.getroot(), encoding="unicode")
    marker = original.find("<score-partwise")
    prologue = original[:marker] if marker != -1 else ""
    path.write_text(prologue + body + "\n")
    return placed
