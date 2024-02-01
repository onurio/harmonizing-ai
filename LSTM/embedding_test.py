import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

from torch.utils.data import DataLoader
import torch.optim as optim

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

device = torch.device("mps" if torch.has_mps else "cpu")

# Load CSV files
input_csv_file = "input_chords.csv"
target_csv_file = "output_chords.csv"

with open(input_csv_file, 'r') as file:
    input_data = file.readlines()

with open(target_csv_file, 'r') as file:
    target_data = file.readlines()

# Split the strings into lists of numbers
input_notes = [line.strip().split(',') for line in input_data]
target_chords = [line.strip().split(',') for line in target_data]


input_notes_flat = [note[0] for note in input_notes]
target_chords_flat = [' '.join(chord) for chord in target_chords]

input_notes_int = [int(note) for note in input_notes_flat]
target_chords_tuples = [tuple(chord.split()) for chord in target_chords_flat]

# Initialize an empty set to store unique notes
unique_notes = set()

# Iterate through chord sequences and add unique notes to the set
for chord_sequence in target_chords:
    unique_notes.update(chord_sequence)

# Convert the set of unique notes back to a list if needed
unique_notes_list = list(unique_notes)

# Print the unique notes
# print(unique_notes_list)


# Find the number of unique MIDI notes
unique_notes = set(input_notes_int)
unique_chord_tuples = set(target_chords_tuples)
num_unique_chords = len(unique_chord_tuples)
num_unique_notes = len(unique_notes_list)

print(f"Number of unique MIDI notes: {num_unique_notes}")
print(f"Number of unique chords: {num_unique_chords}")


note_embedding_dim = 32  # Adjust this dimension based on your data and task
chord_embedding_dim = 32  # Adjust this dimension based on your data and task

hidden_dim = 64  # Adjust this dimension based on your data and task


class MIDIChordDataset(Dataset):
    def __init__(self, notes, chords, unique_notes, padding_value=0):
        self.notes = notes
        self.chords = chords
        self.unique_notes = unique_notes
        self.padding_value = padding_value

        # Create a mapping from note values to indices
        self.note_to_index = {note: index for index,
                              note in enumerate(unique_notes)}

    def __len__(self):
        return len(self.notes)

    def chord_to_indices(self, chord_sequence):
        # Map chord note values to indices using the note_to_index mapping
        return [self.note_to_index.get(note, self.padding_value) for note in chord_sequence]

    def __getitem__(self, idx):
        note_sequence = self.notes[idx]
        chord_sequence = self.chords[idx]

        # Convert sequences to tensors of indices
        note_indices = torch.tensor(
            [self.note_to_index[note] for note in note_sequence], dtype=torch.long)
        chord_indices = torch.tensor(
            self.chord_to_indices(chord_sequence), dtype=torch.long)

        return note_indices, chord_indices

    def collate_fn(self, batch):
        # Calculate the maximum sequence length for chords in the entire batch
        max_chord_length = max(len(chords) for _, chords in batch)

        # Pad notes and chords to the maximum chord length within the entire batch
        padded_notes = []
        padded_chords = []

        for notes, chords in batch:
            pad_notes = torch.cat(
                [notes, torch.zeros(max_chord_length - len(notes), dtype=torch.long)])
            pad_chords = torch.cat([chords, torch.zeros(
                max_chord_length - len(chords), dtype=torch.long)])
            padded_notes.append(pad_notes)
            padded_chords.append(pad_chords)

        padded_notes = torch.stack(padded_notes).to(device)
        padded_chords = torch.stack(padded_chords).to(device)

        return padded_notes, padded_chords


class MIDIChordModel(nn.Module):
    def __init__(self, num_unique_notes, num_unique_chords, embedding_dim, hidden_dim):
        super(MIDIChordModel, self).__init__()
        self.embedding = nn.Embedding(num_unique_notes, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_unique_chords)

    def forward(self, input_notes):
        note_embedded = self.embedding(input_notes)
        out = self.fc1(note_embedded)
        out = self.fc2(out)
        return out


dataset = MIDIChordDataset(input_notes, target_chords,
                           unique_notes=unique_notes_list)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=dataset.collate_fn, shuffle=False)

# Instantiate the model and loss function
model = MIDIChordModel(num_unique_notes, num_unique_chords,
                       note_embedding_dim, hidden_dim)
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop for embedding layer
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for notes, chords in dataloader:  # Assuming you have DataLoader defined
        optimizer.zero_grad()

        # Forward pass
        chord_predictions = model(notes)

        # Reshape chord_predictions and chords to match for loss calculation
        chord_predictions = chord_predictions.view(-1, num_unique_chords)
        chords = chords.view(-1)

        loss = criterion(chord_predictions, chords)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
