import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from LstmEmbeddingModel import MIDIChordModel
import pickle

# Define hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 24
dropout_prob = 0.3
dataset_percent_to_use = 2  # Percentage of data to use for training

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


note_embedding_dim = 64  # Adjust this dimension based on your data and task
# chord_embedding_dim = 32  # Adjust this dimension based on your data and task

hidden_dim1 = 256  # Adjust this dimension based on your data and task
hidden_dim2 = 512


class MIDIChordDataset(Dataset):
    def __init__(self, notes, chords, unique_notes, percent_to_use=100, padding_value=0):
        self.notes = notes
        self.chords = chords
        self.unique_notes = unique_notes
        self.percent_to_use = percent_to_use  # Percentage of data to use
        self.padding_value = padding_value

        # Create a mapping from note values to indices
        self.note_to_index = {note: index for index,
                              note in enumerate(unique_notes)}

        # Determine the number of samples to use based on the percentage
        num_samples_to_use = len(notes) * percent_to_use // 100

        # Randomly select a subset of data
        random.seed(42)  # Set a random seed for reproducibility
        self.sample_indices = random.sample(
            range(len(notes)), num_samples_to_use)

    def __len__(self):
        return len(self.sample_indices)  # Use the selected sample indices

    def chord_to_indices(self, chord_sequence):
        # Map chord note values to indices using the note_to_index mapping
        return [self.note_to_index.get(note, self.padding_value) for note in chord_sequence]

    def __getitem__(self, idx):
        # Use the selected sample indices
        sample_idx = self.sample_indices[idx]

        note_sequence = self.notes[sample_idx]
        chord_sequence = self.chords[sample_idx]

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


dataset = MIDIChordDataset(input_notes, target_chords,
                           unique_notes=unique_notes_list, percent_to_use=dataset_percent_to_use)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=dataset.collate_fn, shuffle=False)

# Instantiate the model and loss function
model = MIDIChordModel(num_unique_notes,
                       note_embedding_dim, hidden_dim1, hidden_dim2, dropout_prob)
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

graph = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for notes, chords in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()

        # Forward pass
        chord_predictions = model(notes)

        # Reshape chord_predictions and chords to match for loss calculation
        chord_predictions = chord_predictions.view(-1, num_unique_notes)
        chords = chords.view(-1)

        loss = criterion(chord_predictions, chords)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    graph.append(average_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

# Save the model's state dictionary to a file
torch.save(model.state_dict(), 'model_weights.pth')
index_to_note = {index: note for note, index in dataset.note_to_index.items()}
# Save the "note to index" dictionary to a file after training
with open('note_to_index.pkl', 'wb') as file:
    pickle.dump(dataset.note_to_index, file)

# Save other relevant information such as hyperparameters
model_info = {
    'num_unique_notes': num_unique_notes,
    'num_unique_chords': num_unique_chords,
    'note_embedding_dim': note_embedding_dim,
    'hidden_dim1': hidden_dim1,
    'hidden_dim2': hidden_dim2,
    'dropout_prob': dropout_prob
}
torch.save(model_info, 'model_info.pth')


# Set the model to evaluation mode
model.eval()

# Define the initial input (note) to start the generation
initial_input = torch.tensor([64], dtype=torch.long).unsqueeze(
    0).to(device)  # Adjust your_initial_note as needed

# Number of chords to generate
num_chords_to_generate = 1  # Adjust as needed

# Initialize a list to store the generated chord sequence
generated_chord_sequences = []

# Generate chord sequences
with torch.no_grad():
    current_input = initial_input
    for _ in range(num_chords_to_generate):
        # Get the model's prediction for the next chord
        chord_predictions = model(current_input)

        print(chord_predictions, chord_predictions.size())

        # Get the chord prediction for the last time step
        chord_prediction = chord_predictions[0, -1]

        # Create a list to store note-probability pairs
        note_probabilities = []

        # Iterate through the chord prediction tensor
        for note_index in range(chord_prediction.size(0)):
            # Get the probability for the current note
            probability = chord_prediction[note_index].item()

            # Append the note-probability pair to the list
            note_probabilities.append((note_index, probability))

        # Sort the note-probability pairs by probability in descending order
        note_probabilities.sort(key=lambda x: x[1], reverse=True)

        # Print the note-probability pairs
        for note_index, probability in note_probabilities:
            # Convert note index to the corresponding note value (e.g., MIDI note pitch)
            note_value = unique_notes_list[note_index]

            # Print the note value and its probability
            print(f"Note Value: {note_value}, Probability: {probability:.5f}")

        # Sample the next chord index based on the predictions
        next_chord_index = torch.multinomial(
            torch.sigmoid(chord_predictions)[0, -1], 1).item()

        # Append the next chord index to the generated sequence
        generated_chord_sequences.append(next_chord_index)

        # Prepare the input for the next step
        current_input = torch.tensor(
            [[next_chord_index]], dtype=torch.long).to(device)
