import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the ChordPredictor model with embeddings


class ChordPredictor(nn.Module):
    def __init__(self, note_input_size, chord_input_size, embedding_dim, hidden_size, output_size, num_layers=2):
        super(ChordPredictor, self).__init__()
        self.note_embedding = nn.Embedding(note_input_size, embedding_dim)
        self.chord_embedding = nn.Embedding(chord_input_size, embedding_dim)
        self.note_lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.chord_lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * (4 + 1),
                            output_size)  # 4 chords + 1 note

    def forward(self, note_input, chord_input):
        note_embedded = self.note_embedding(note_input)
        chord_embedded = self.chord_embedding(chord_input)

        note_output, _ = self.note_lstm(note_embedded)
        chord_output, _ = self.chord_lstm(chord_embedded)

        # Take the output of the last time step for both note and chord LSTMs
        note_output = note_output[:, -1, :]
        chord_output = chord_output[:, -1, :]

        # Concatenate the outputs of note and chord LSTMs
        combined_output = torch.cat((note_output, chord_output), dim=1)

        # Pass the combined output through a fully connected layer
        output = self.fc(combined_output)

        return output


# Read CSV data as a list of lists
input_df = []
with open('input_chords.csv', 'r') as file:
    for line in file:
        row = [int(value) for value in line.strip().split(',')]
        input_df.append(row)

output_df = []
with open('output_chords.csv', 'r') as file:
    for line in file:
        row = [int(value) for value in line.strip().split(',')]
        output_df.append(row)

# Convert list of lists to PyTorch tensors
note_sequences = torch.tensor(input_df, dtype=torch.long).to(device)
chord_sequences = torch.tensor(output_df, dtype=torch.long).to(device)

# Hyperparameters
note_input_size = 128  # Assuming MIDI representation typically ranges from 0 to 127
chord_input_size = 128  # Assuming MIDI representation for chords
embedding_dim = 32  # You may adjust this based on your preference and experimentation
hidden_size = 64
output_size = 128  # MIDI representation for the next chord
learning_rate = 0.001
num_epochs = 10
window_size = 5  # Including the last 4 chords and the current note

# Prepare sequences with sliding window
input_sequences = []
target_chords = []

for i in range(window_size, len(chord_sequences)):
    print(i, end='\r')
    # The last 4 chords
    last_four_chords = chord_sequences[i - window_size:i - 1]

    # The current note
    current_note = note_sequences[i]

    # Concatenate to form the input sequence
    input_sequence = torch.cat(
        (last_four_chords.view(-1), current_note.view(-1)), dim=0)

    # Target chord
    target_chord = chord_sequences[i]

    input_sequences.append(input_sequence)
    target_chords.append(target_chord)


note_sequences_windowed = torch.stack(input_sequences)
target_chords_windowed = torch.stack(target_chords)


print('finished preparing data')

# Create an instance of the ChordPredictor model
model = ChordPredictor(note_input_size, chord_input_size,
                       embedding_dim, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    output = model(note_sequences_windowed, note_sequences_windowed)

    # Reshape output and target to fit the cross-entropy loss
    output = output.view(-1, output_size)
    target_chords_windowed = target_chords_windowed.view(-1)

    # Compute the loss
    loss = criterion(output, target_chords_windowed)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the trained model for chord prediction with the specified context.
