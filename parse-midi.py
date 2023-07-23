import mido
import csv
import numpy as np
import os

file_path = "Liebestraume.mid"
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
        for chord in chords:
            one_hot = np.zeros(127)
            notes = chord['notes']
            notes.sort()
            notes.reverse()
            one_hot[notes[0]] = 1.
            one_hot = one_hot.tolist()
            one_hot.insert(0, round(chord['position'], 3))
            writer.writerow(one_hot)

    with open('output_'+file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for chord in chords:
            one_hot = np.zeros(127)
            for ch in chord:
                notes = chord['notes']
                for note in notes:
                    one_hot[note] = 1.
            one_hot = one_hot.tolist()
            writer.writerow(one_hot)


chords = []


def parse_midi_file(file_path, time_signature):
    print("Parsing MIDI file:", file_path)
    try:
        mid = mido.MidiFile(file_path)
        print("MIDI file type:", mid.type)
        print("Ticks per beat:", mid.ticks_per_beat)

        # Dictionary to store active notes
        active_notes = {}
        absoluteTime = 0
        for i, track in enumerate(mid.tracks):
            for msg in track:
                absoluteTime += msg.time
                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        # Note-on event: store the note's start time and position
                        position = calculate_position(
                            absoluteTime, mid.ticks_per_beat, time_signature)
                        active_notes[msg.note] = {
                            'start': absoluteTime, 'end': None, 'position': position}

                elif msg.type == 'note_off':
                    # Note-off event: update the note's end time and calculate position
                    if msg.note in active_notes:
                        del active_notes[msg.note]

                elif msg.type == 'time_signature':
                    time_signature = msg.dict()

                elif msg.type == 'track_name':
                    track_name = msg.name
                    print("Track name:", track_name)

                currentNotes = len(active_notes)
                if currentNotes > 1 and currentNotes < 5:
                    currentNotesInChord = []
                    lastNoteTimes = None
                    for note, times in active_notes.items():
                        currentNotesInChord.append(note)
                        lastNoteTimes = times
                    chords.append({'notes': currentNotesInChord,
                                  'position': lastNoteTimes['position']})

    except mido.InvalidMidiDataError as e:
        print("Invalid MIDI file:", e)


# parse_midi_file(file_path, time_signature)

# loop over midi folder
for file in os.listdir("midi"):
    if file.endswith(".mid"):
        parse_midi_file('./midi/'+file, time_signature)

# Write simultaneous notes to CSV file
# print(chords)
csv_file_path = 'chords.csv'
print(csv_file_path)
write_chords_to_csv(chords, csv_file_path)
# print("CSV file written successfully:", csv_file_path)
