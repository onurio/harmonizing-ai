import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class ChordPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ChordPredictionModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape input tensor to have a batch dimension
        x = x.unsqueeze(0)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.linear(out[:, -1, :])
        return torch.sigmoid(out)

# Training loop


def train_model(model, data_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


# Set hyperparameters
input_dim = 128
hidden_dim = 256
output_dim = 127
num_layers = 8
num_epochs = 100
batch_size = 1
learning_rate = 0.01  # Adjust the learning rate as desired

# Initialize the model, loss, and optimizer
model = ChordPredictionModel(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Define your training dataset


class ChordDataset(Dataset):
    def __init__(self, input_file, target_file):
        self.input_data = pd.read_csv(input_file).values.tolist()
        self.target_data = pd.read_csv(target_file).values.tolist()

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        input_data = torch.FloatTensor(self.input_data[index])
        target = torch.FloatTensor(self.target_data[index])
        return input_data, target


# Load input and target data from separate CSV files
# Replace with the path to your input CSV file
input_csv_file = 'input_chords.csv'
# Replace with the path to your target CSV file
target_csv_file = 'output_chords.csv'

# Create the dataset
dataset = ChordDataset(input_csv_file, target_csv_file)

# Create a data loader for batching your training data
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)


# Generate an example input tensor
example_input = torch.tensor([0.922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Replace with your desired input

example_input = example_input.reshape(1, 128)

# Pass the input through the model to get the prediction
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    output = model(example_input)

# Process the output as desired
# For example, you can convert the output tensor to a numpy array
output_array = output.detach().numpy()
# Print the generated example output
print("Generated Example Output:")
print(output_array)
