import torch.onnx
import torch
from model import MusicTransformer
import numpy as np

loaded_dict = torch.load('model_state_dict.pth',map_location=torch.device('cpu'))
model_state_dict = loaded_dict['model_state_dict']
loaded_hyperparameters = loaded_dict['hyperparameters']

device = torch.device("cpu")

learning_rate = loaded_hyperparameters['learning_rate']
batch_size = loaded_hyperparameters['batch_size']
embed_size = loaded_hyperparameters['embed_size']
num_layers = loaded_hyperparameters['num_layers']
heads = loaded_hyperparameters['heads']
forward_expansion = loaded_hyperparameters['forward_expansion']
dropout = loaded_hyperparameters['dropout']
max_length = loaded_hyperparameters['max_length']
note_vocab_size = loaded_hyperparameters['note_vocab_size']

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
model.eval()

dummy_input = [0,0,0,0,0,0,0,0,0,60,0,0]
dummy_input = np.array(dummy_input)
dummy_input = torch.tensor(dummy_input, dtype=torch.long).reshape(1,4,3).to(device)

# Export the model to ONNX
torch.onnx.export(model, (dummy_input, None), "model.onnx", opset_version=12, export_params=True)
