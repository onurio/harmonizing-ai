import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Step 1: Read the CSV file
data = pd.read_csv('output_chords.csv').values.tolist()
print(data[0:10])

# Step 2: Determine maximum sequence length
max_seq_length = max(data.apply(lambda row: len(row), axis=1))

# Step 3: Tokenize categorical values
category_to_index = {}  # Create a dictionary to map categories to indices
index = 1  # Start index from 1, reserve 0 for padding
for category in data.stack().unique():
    category_to_index[category] = index
    index += 1

# Step 4: Pad sequences with 0s
padded_sequences = []
for row in data.values:
    padded_sequence = [category_to_index.get(category, 0) for category in row]
    padded_sequence += [0] * (max_seq_length - len(padded_sequence))
    padded_sequences.append(padded_sequence)

# Step 5: Create a PyTorch Dataset and DataLoader


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = MyDataset(padded_sequences)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 6: Define your embedding layer
vocab_size = len(category_to_index) + 1  # Add 1 for padding
embedding_dim = 32
embedding = torch.nn.Embedding(vocab_size, embedding_dim)

# Step 7: Iterate through the dataset and pass through the embedding layer
for batch in dataloader:
    # Convert batch to tensor
    batch_tensor = torch.tensor(batch)

    # Pass through the embedding layer
    embedded_batch = embedding(batch_tensor)

    # Now, 'embedded_batch' contains the embedded representations of your sequences
