import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pythonosc import osc_server
from pythonosc import dispatcher
from pythonosc import udp_client
import argparse

# device = torch.device("cuda")


class ChordPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob):
        super(ChordPredictionModel, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim1, 256)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.dense2 = nn.Linear(256, output_dim)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dense1(out)
        out = self.dropout3(out)
        out = self.dense2(out)
        return out


# Set hyperparameters
input_dim = 128
hidden_dim1 = 256
hidden_dim2 = 512
output_dim = 127
dropout_prob = 0.5

# Initialize the model
model = ChordPredictionModel(
    input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob)


model.load_state_dict(torch.load('path_to_save_model.pth',
                      map_location=torch.device('cpu')))


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
parser.add_argument("--port", type=int, default=9001,
                    help="The port the OSC server is listening on")
args = parser.parse_args()
# Set up OSC client (for sending messages)
client = udp_client.SimpleUDPClient(args.ip, args.port)
client.send_message("/status", 1)


def handle_message(unused_addr, *values):
    # Convert the received values to a PyTorch tensor
    # input_data = torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # unsqueeze(0) to add batch dimension
    print(f"/input - {values}")
    # Send the input data to the model and get the output
    indices = np.array(values[1:])
    input_arr = np.zeros(127)
    input_arr[indices] = 1
    input_arr = np.insert(input_arr, 0, values[:1])
    print(f" did it work?? {input_arr}")

    example_input = torch.tensor(
        input_arr, dtype=torch.float32).reshape(1, 128)
    output = model(example_input)
    # Convert the output to a list and send it back over OSC
    # squeeze(0) to remove batch dimension
    output_list = output.sigmoid().squeeze(0).tolist()
    print(f"the output = {output_list}")
    client.send_message("/prediction", output_list)


# Set up OSC server (for receiving messages)
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/input", handle_message)


try:
    server = osc_server.ThreadingOSCUDPServer(('localhost', 9000), dispatcher)
    print("Serving on {}".format(server.server_address))
    client.send_message("/status", 2)
    server.serve_forever()
except KeyboardInterrupt:
    print("\nShutting down OSC Server...")
    server.shutdown()
    server.server_close()
    client.send_message("/status", 0)
    print("OSC Server shut down successfully.")
