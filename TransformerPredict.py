import torch
import numpy as np
from pythonosc import osc_server
from pythonosc import dispatcher
from pythonosc import udp_client
import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from model import MusicTransformer


inputPort = 9515
outputPort = 9516

# device = torch.device("cuda") 

device = torch.device("cpu")


loaded_dict = torch.load('model_state_dict.pth',map_location=torch.device('cpu'))
model_state_dict = loaded_dict['model_state_dict']
loaded_hyperparameters = loaded_dict['hyperparameters']


learning_rate = loaded_hyperparameters['learning_rate']
batch_size = loaded_hyperparameters['batch_size']
embed_size = loaded_hyperparameters['embed_size']
num_layers = loaded_hyperparameters['num_layers']
heads = loaded_hyperparameters['heads']
forward_expansion = loaded_hyperparameters['forward_expansion']
dropout = loaded_hyperparameters['dropout']
max_length = loaded_hyperparameters['max_length']
note_vocab_size = loaded_hyperparameters['note_vocab_size']

print(device)
model = MusicTransformer(
    note_vocab_size,
    embed_size,
    num_layers,
    heads,
    device,
    forward_expansion,
    dropout,
    max_length
).to(device)

# Initialize the model
model.load_state_dict(model_state_dict)
model.to(device)


# Pass the input through the model to get the prediction

model.eval()  # Set the model to evaluation mode


parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
                    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=outputPort,
                    help="The port the OSC server is listening on")
args = parser.parse_args()
# Set up OSC client (for sending messages)
client = udp_client.SimpleUDPClient(args.ip, args.port)
client.send_message("/status", 1)


last_three_chords = [0,0,0,0,0,0,0,0,0]


def handle_message(_, value):
    # Convert the received values to a PyTorch tensor
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
    print(top_indices,value)
    last_three_chords = last_three_chords[3:]
    last_three_chords.extend(top_indices)
    print(last_three_chords)
    client.send_message("/prediction",top_indices)


# Set up OSC server (for receiving messages)
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/input", handle_message)


try:
    server = osc_server.ThreadingOSCUDPServer(('localhost', inputPort), dispatcher)
    print("Serving on {}".format(server.server_address))
    client.send_message("/status", 2)
    server.serve_forever()
except KeyboardInterrupt:
    print("\nShutting down OSC Server...")
    server.shutdown()
    server.server_close()
    client.send_message("/status", 0)
    print("OSC Server shut down successfully.")
