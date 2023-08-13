
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from ChordPredictionModel import ChordPredictionModelLightning
from lightning import lightning as L


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


# Train the model
if __name__ == '__main__':
    # Set hyperparameters
    input_dim = 128
    hidden_dim1 = 256
    hidden_dim2 = 512
    output_dim = 128
    learning_rate = 0.005  # Adjust the learning rate as desired
    dropout_prob = 0.3
    batch_size = 512
    num_epochs = 10

    criterion = nn.CrossEntropyLoss()

    # Initialize the model
    model = ChordPredictionModelLightning(
        input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob, learning_rate, criterion)
    trainer = L.Trainer(accelerator='gpu', devices=1,
                        max_epochs=num_epochs, profiler="simple")

    # Load input and target data from separate CSV files
    # Replace with the path to your input CSV file
    input_csv_file = 'input_chords.csv'
    # Replace with the path to your target CSV file
    target_csv_file = 'output_chords.csv'

    dataset = ChordDataset(input_csv_file, target_csv_file)

    # Create a data loader for batching your training data
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    trainer.fit(model, train_dataloaders=train_loader)

# save_path = "path_to_save_model.pth"
# torch.save(model.state_dict(), save_path)


# # Generate an example input tensor
# example_input = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Replace with your desired input

# example_input = example_input.reshape(1, 128).to(device).sigmoid()

# # Pass the input through the model to get the prediction
# with torch.no_grad():
#     model.eval()  # Set the model to evaluation mode
#     output = model(example_input)

# # Process the output as desired
# # For example, you can convert the output tensor to a numpy array
# output_array = output.detach().cpu().numpy().round(5)

# # Print the generated example output
# print("Generated Example Output:")
# print(output_array)
