
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset

device = torch.device("mps")


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
learning_rate = 0.0005  # Adjust the learning rate as desired
dropout_prob = 0.5
batch_size = 16
num_epochs = 300

# Initialize the model
model = ChordPredictionModel(
    input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob)

model.to(device)


# Training loop


def train_model(model, data_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            # Move data to GPU if available
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(train_loader)

        # Print the progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

# Initialize the model, loss, and optimizer


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

save_path = "path_to_save_model.pth"
torch.save(model.state_dict(), save_path)


# Generate an example input tensor
example_input = torch.tensor([0.922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Replace with your desired input

example_input = example_input.reshape(1, 128).to(device).sigmoid()

# Pass the input through the model to get the prediction
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    output = model(example_input)

# Process the output as desired
# For example, you can convert the output tensor to a numpy array
output_array = output.detach().cpu().numpy()

# Print the generated example output
print("Generated Example Output:")
print(output_array)
