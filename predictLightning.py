import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pythonosc import osc_server
from pythonosc import dispatcher
from pythonosc import udp_client
import argparse
from ChordPredictionModel import ChordPredictionModelLightning


device = torch.device('cpu')
# # Set hyperparameters
# input_dim = 128
# hidden_dim1 = 256
# hidden_dim2 = 512
# output_dim = 128
# dropout_prob = 0.3

criterion = nn.MSELoss()

# Initialize the model
checkpoint_path = '2023-08-22T14:18:55.826048MSE.ckpt'
model = ChordPredictionModelLightning.load_from_checkpoint(
    checkpoint_path=checkpoint_path, criterion=criterion)

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
parser.add_argument("--port", type=int, default=9005,
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

    example_input = torch.tensor(
        input_arr, dtype=torch.float32).reshape(1, 128).to(device)
    output = model(example_input)
    # Convert the output to a list and send it back over OSC
    # squeeze(0) to remove batch dimension
    output_list = output.squeeze(0).tolist()
    min = np.min(output_list)
    output_list = np.subtract(output_list, min)
    max = np.max(output_list)
    output_list = np.divide(output_list, max)
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
