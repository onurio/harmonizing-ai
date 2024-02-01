def create_note_vocabulary(num_octaves=7, start_octave=0):
    """
    Create a vocabulary of notes represented as strings (e.g., 'C4').

    Args:
    - num_octaves (int): The number of octaves to include in the vocabulary.
    - start_octave (int): The starting octave number.

    Returns:
    - vocabulary (list of strings): The note vocabulary.
    """
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E',
                     'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    vocabulary = []

    for octave in range(start_octave, start_octave + num_octaves):
        for pitch_class in pitch_classes:
            note = f'{pitch_class}{octave}'
            vocabulary.append(note)

    return vocabulary


def midi_to_pitch(midi_note_str):
    """
    Convert a MIDI note number represented as a string to a pitch representation (e.g., 'C4', 'D#5').

    Args:
    - midi_note_str (str): The MIDI note number as a string in the range 1 - 127.

    Returns:
    - pitch_str (str): The pitch representation (e.g., 'C4').
    """
    # Convert the input MIDI note string to an integer
    try:
        midi_note_int = int(midi_note_str)
    except ValueError:
        raise ValueError(
            "Invalid MIDI note string. It should be an integer between 1 and 127.")

    # Check if the MIDI note number is within the valid range (1 - 127)
    if not (1 <= midi_note_int <= 127):
        raise ValueError(
            "MIDI note number is out of range. It should be between 1 and 127.")

    # Define a list of pitch classes
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E',
                     'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Calculate the pitch class (0 - 11) and octave (0 - 9) from the MIDI note number
    pitch_class_index = (midi_note_int - 12) % 12
    octave = (midi_note_int - 12) // 12

    # Create the pitch representation string
    pitch_str = f'{pitch_classes[pitch_class_index]}{octave}'

    return pitch_str
