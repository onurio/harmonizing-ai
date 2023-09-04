# Import the function to be tested
import pytest
from notevocab import midi_to_pitch


def test_valid_midi_notes():
    # Test valid MIDI note numbers within the range 1 - 127
    test_cases = {
        '60': 'C4',
        '64': 'E4',
        '69': 'A4',
        '72': 'C5',
        '81': 'A5',
        '127': 'G9',
    }

    for midi_note, expected_pitch in test_cases.items():
        result = midi_to_pitch(midi_note)
        assert result == expected_pitch


def test_invalid_midi_notes():
    # Test invalid MIDI note numbers
    invalid_midi_notes = ['-1', '0', '128', 'abc', '65.5']

    for midi_note in invalid_midi_notes:
        with pytest.raises(ValueError):  # Expecting a ValueError
            midi_to_pitch(midi_note)
