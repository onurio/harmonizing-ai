import csv
import numpy as np
import os
import copy
from mido import MidiFile

midi_filename = "./midi/bach-tempered-claviere-fugue1.mid"
time_signature = {}


def calculate_position(time, ticks_per_beat, time_signature):
    # Numerator usually represents beats per bar
    beats_per_bar = time_signature['numerator']

    # We calculate how many ticks are in a beat, then multiply by the number of beats per bar.
    ticks_per_bar = ticks_per_beat * beats_per_bar

    # We calculate the position in the bar by taking the current time modulo the number of ticks in a bar.
    # This gives us the current tick position within the bar.
    # Dividing by the number of ticks in a bar normalizes it to a value between 0 and 1.
    position = (time % ticks_per_bar) / ticks_per_bar

    return position


def write_chords_to_csv(chords, file_path):
    with open('input_'+file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for (time, chord) in chords:
            notes = chord
            notes.sort()
            notes.reverse()
            writer.writerow([int(notes[0])])

    with open('output_'+file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for (time, chord) in chords:
            writer.writerow([int(num) for num in chord])


def get_time_signature(midi_file):
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                return msg.numerator, msg.denominator

    return None, None


def extract_chords_from_midi(midi_filename):
    try:
        midi_file = MidiFile(midi_filename)
        numerator, denominator = get_time_signature(midi_file)

        chords = []

        current_chord = set()
        current_time = 0
        microseconds_per_beat = 500000  # Default value

        for msg in midi_file:

            if msg.type == 'note_off':
                current_chord.discard(msg.note)
            if msg.type == 'note_on':
                # Convert microseconds to seconds
                current_time += msg.time * microseconds_per_beat

                if msg.velocity > 0:
                    current_chord.add(msg.note)
                else:
                    current_chord.discard(msg.note)

                if current_chord:
                    note_amount = len(current_chord)
                    note_list = list(current_chord)
                    if note_amount <= 6:
                        note_list.extend([0] * (6 - note_amount))
                    if note_amount > 1 and note_amount <= 6:
                        chords.append((current_time, note_list))

        return chords, (numerator, denominator)
    except Exception as e:
        print("Error parsing MIDI file:", midi_filename, e)
        return [], None


def time_in_bar(time, time_signature):
    beats_per_bar = time_signature[0]
    beat_type = time_signature[1]

    time_in_beats = time / (60 / beat_type)
    time_fraction = (time_in_beats % beats_per_bar) / beats_per_bar

    return time_fraction


total_chords = []
# # loop over midi folder


def traverse_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(file)
            if file.endswith(".mid") | file.endswith("midi"):

                file_path = os.path.join(root, file)
                extracted_chords, time_signature = extract_chords_from_midi(
                    file_path)
                total_chords.extend(extracted_chords)


traverse_directory('./midi')

# for file in os.listdir("midi"):
#     chords = []
#     if file.endswith(".mid"):
#         print(file)

# # Write simultaneous notes to CSV file
# # print(chords)
csv_file_path = 'chords.csv'
# print(csv_file_path)
write_chords_to_csv(total_chords, csv_file_path)
# # print("CSV file written successfully:", csv_file_path)


# with open('just_chords', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for (time, chord) in total_chords:
#         writer.writerow(chord.sort())
