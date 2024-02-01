import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pythonosc import osc_server
from pythonosc import dispatcher
from pythonosc import udp_client
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query * keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just a way to do batch matrix multiplication
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = FeedForward(embed_size, forward_expansion * embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class MusicTransformer(nn.Module):
    def __init__(self, chord_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(MusicTransformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(chord_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        # Output layer to predict two notes (assuming each note is represented by a single number)
        self.fc_out = nn.Linear(embed_size, 128)

    def forward(self, x, mask):
        N, sequence_length, chord_size = x.size()

        x = x.view(N, sequence_length * chord_size)
        positions = torch.arange(0, sequence_length * chord_size).expand(N, sequence_length * chord_size).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)

        # Use the output of the first token for classification or apply pooling
        out = out.mean(dim=1)  # Average pooling over the sequence

        out = self.fc_out(out)
        return out

# device = torch.device("cuda") 

device = torch.device("cpu")

# Assuming the definition of the MusicTransformer model is available from model.py
# Initialize the Music Transformer model
chord_vocab_size = 10950  # As specified earlier
embed_size = 1024  # Example size, adjust as needed
num_layers = 8  # Example value, adjust as needed
heads = 4  # Example value, adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forward_expansion = 4  # Example value, adjust as needed
dropout = 0.3  # Example dropout rate, adjust as needed
max_length = 100  # Maximum sequence length, adjust as needed

print(device)
model = MusicTransformer(
    chord_vocab_size,
    embed_size,
    num_layers,
    heads,
    device,
    forward_expansion,
    dropout,
    max_length
).to(device)

# Initialize the model
model.load_state_dict(torch.load('model_state_dict.pth',map_location=torch.device('cpu')))
model.to(device)



# Generate an example input tensor
# example_input = torch.tensor([0.922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Replace with your desired input

# example_input = example_input.reshape(1, 128)

# Pass the input through the model to get the prediction

model.eval()  # Set the model to evaluation mode

# for i in range(10):
#     output = model(example_input)

#     # Process the output as desired
#     # For example, you can convert the output tensor to a numpy array
#     output_array = output.sigmoid().detach().numpy()
#     ready_out = output_array * (output_array > 0.7)
#     ready_out = np.insert(ready_out, 0, [i/10.])
#     # Print the generated example output
#     print("Generated Example Output:")
#     print(ready_out)
#     example_input = torch.tensor(ready_out).reshape(1, 128)


parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
                    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=9518,
                    help="The port the OSC server is listening on")
args = parser.parse_args()
# Set up OSC client (for sending messages)
client = udp_client.SimpleUDPClient(args.ip, args.port)
client.send_message("/status", 1)


last_three_chords = [0,0,0,0,0,0,0,0,0]


def handle_message(unused_addr, value):
    # Convert the received values to a PyTorch tensor
    # input_data = torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # unsqueeze(0) to add batch dimension
    global last_three_chords  # Declare last_three_chords as global

    new_input = last_three_chords + [value,0,0]

    print(f"/input - {new_input}")
    # Send the input data to the model and get the output
    input_arr = np.array(new_input)
    input = torch.tensor(
        input_arr, dtype=torch.long).reshape(1,4,3).to(device)

    with torch.no_grad():
        output = model(input, None)  # Assuming no mask for simplicity
        probabilities = torch.softmax(output, dim=-1)
        top_probabilities, top_indices = torch.topk(probabilities, 2, dim=-1)
        top_indices = top_indices.cpu().numpy().flatten().tolist()  # Move back to CPU and flatten

    top_indices.insert(0,value) # add the last note input.
    last_three_chords = last_three_chords[3:]
    last_three_chords.extend(top_indices)
    print(last_three_chords)
    client.send_message("/prediction",top_indices)


# Set up OSC server (for receiving messages)
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/input", handle_message)


try:
    server = osc_server.ThreadingOSCUDPServer(('localhost', 9514), dispatcher)
    print("Serving on {}".format(server.server_address))
    client.send_message("/status", 2)
    server.serve_forever()
except KeyboardInterrupt:
    print("\nShutting down OSC Server...")
    server.shutdown()
    server.server_close()
    client.send_message("/status", 0)
    print("OSC Server shut down successfully.")
