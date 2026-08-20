"""Tests for muscriptor/utils/harmony.py (chord symbols in a MusicXML score).

MuseScore is never invoked: the scores here are written by hand, which is also
the only way to pin down the exact placement the engraver will read back.
"""

import xml.etree.ElementTree as ET

from muscriptor.utils.harmony import add_chord_symbols

PROLOGUE = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">\n'
)


def _score(measures, divisions=1, parts=1):
    """A partwise score. `measures` is a list of per-measure note durations."""
    root = ET.Element("score-partwise", version="4.0")
    part_list = ET.SubElement(root, "part-list")
    for number in range(1, parts + 1):
        ET.SubElement(part_list, "score-part", id=f"P{number}")
    for number in range(1, parts + 1):
        part = ET.SubElement(root, "part", id=f"P{number}")
        for index, durations in enumerate(measures, start=1):
            measure = ET.SubElement(part, "measure", number=str(index))
            if index == 1:
                attributes = ET.SubElement(measure, "attributes")
                ET.SubElement(attributes, "divisions").text = str(divisions)
            for duration in durations:
                note = ET.SubElement(measure, "note")
                pitch = ET.SubElement(note, "pitch")
                ET.SubElement(pitch, "step").text = "C"
                ET.SubElement(pitch, "octave").text = "4"
                ET.SubElement(note, "duration").text = str(duration)
    return ET.ElementTree(root)


def _write(tmp_path, measures, **kwargs):
    path = tmp_path / "score.musicxml"
    body = ET.tostring(_score(measures, **kwargs).getroot(), encoding="unicode")
    path.write_text(PROLOGUE + body)
    return path


def _harmonies(path):
    """(measure number, root as written, kind, offset) for each chord symbol."""
    found = []
    for measure in ET.parse(path).getroot().find("part").findall("measure"):
        for harmony in measure.findall("harmony"):
            step = harmony.findtext("root/root-step")
            alter = harmony.findtext("root/root-alter")
            found.append(
                (
                    measure.get("number"),
                    step + {None: "", "1": "#", "-1": "b"}[alter],
                    harmony.findtext("kind"),
                    harmony.findtext("offset"),
                )
            )
    return found


def test_symbols_land_in_the_right_measure(tmp_path):
    path = _write(tmp_path, [[1, 1, 1, 1]] * 3)
    assert add_chord_symbols(path, [(0.0, "C"), (4.0, "Am7"), (8.0, "F")]) == 3
    assert _harmonies(path) == [
        ("1", "C", "major", None),
        ("2", "A", "minor-seventh", None),
        ("3", "F", "major", None),
    ]


def test_a_symbol_precedes_the_note_it_belongs_to(tmp_path):
    """The engraver reads position from the order, so a harmony goes first."""
    path = _write(tmp_path, [[1, 1, 1, 1]])
    add_chord_symbols(path, [(2.0, "G")])
    measure = ET.parse(path).getroot().find("part").find("measure")
    tags = [child.tag for child in measure]
    assert tags == ["attributes", "note", "note", "harmony", "note", "note"]


def test_a_change_over_a_held_note_gets_an_offset(tmp_path):
    """Nothing is attacked at beat 3, so the chord hangs off the note at beat 1."""
    path = _write(tmp_path, [[4]], divisions=2)  # one whole note, 2 divisions/quarter
    add_chord_symbols(path, [(1.0, "D")])
    assert _harmonies(path) == [("1", "D", "major", "2")]


def test_accidentals_and_no_chord_are_spelled_out(tmp_path):
    path = _write(tmp_path, [[1, 1, 1, 1]] * 2)
    add_chord_symbols(path, [(0.0, "Bb7"), (1.0, "F#m7b5"), (4.0, "N.C.")])
    assert _harmonies(path) == [
        ("1", "Bb", "dominant", None),
        # Beat 2 is attacked, so this one needs no offset — it is written
        # straight onto the note.
        ("1", "F#", "half-diminished", None),
        ("2", "C", "none", None),
    ]
    # "No chord" prints no root letter, only the N.C. text.
    harmony = (
        ET.parse(path).getroot().find("part").findall("measure")[1].find("harmony")
    )
    assert harmony.find("root/root-step").get("text") == ""
    assert harmony.find("kind").get("text") == "N.C."


def test_symbols_past_the_last_bar_line_are_dropped(tmp_path):
    path = _write(tmp_path, [[1, 1, 1, 1]])
    assert add_chord_symbols(path, [(0.0, "C"), (4.0, "G"), (99.0, "D")]) == 1
    assert _harmonies(path) == [("1", "C", "major", None)]


def test_only_the_top_part_carries_the_symbols(tmp_path):
    """Repeating the chords over every staff would be noise, not information."""
    path = _write(tmp_path, [[1, 1, 1, 1]], parts=3)
    add_chord_symbols(path, [(0.0, "C")])
    counts = [
        len(part.findall("measure/harmony"))
        for part in ET.parse(path).getroot().findall("part")
    ]
    assert counts == [1, 0, 0]


def test_the_doctype_survives(tmp_path):
    """MusicXML readers dispatch on it, and ElementTree does not write one."""
    path = _write(tmp_path, [[1, 1, 1, 1]])
    add_chord_symbols(path, [(0.0, "C")])
    assert path.read_text().startswith(PROLOGUE)


def test_a_score_with_no_part_is_left_alone(tmp_path):
    path = tmp_path / "score.musicxml"
    path.write_text(PROLOGUE + "<score-partwise><part-list /></score-partwise>")
    assert add_chord_symbols(path, [(0.0, "C")]) == 0
