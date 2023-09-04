
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from ChordPredictionModel import ChordPredictionModelLightning
from lightning import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from datetime import datetime
import random


class ChordDataset(Dataset):
    def __init__(self, input_file, target_file, subset_percentage=100):
        self.input_data = pd.read_csv(input_file).values.tolist()
        self.target_data = pd.read_csv(target_file).values.tolist()
        self.subset_size = int(len(self.input_data) * subset_percentage / 100)

    def __len__(self):
        return self.subset_size

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
    learning_rate = 0.001  # Adjust the learning rate as desired
    dropout_prob = 0.3
    batch_size = 256
    num_epochs = 10

    # first_criterion = nn.BCEWithLogitsLoss()
    # second_criterion = nn.BCELoss()
    # third_criterion = nn.MSELoss()
    forth_criterion = nn.CrossEntropyLoss()

    criterions = [forth_criterion]

    for criterion in criterions:
        # Initialize the model
        model = ChordPredictionModelLightning(
            input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob, learning_rate, criterion, False)
        trainer = L.Trainer(accelerator='gpu', devices=1,
                            max_epochs=num_epochs, profiler="simple", callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)])

        tuner = Tuner(trainer)

        # Load input and target data from separate CSV files
        # Replace with the path to your input CSV file
        input_csv_file = 'input_chords.csv'
        # Replace with the path to your target CSV file
        target_csv_file = 'output_chords.csv'

        dataset = ChordDataset(input_csv_file, target_csv_file, 100)

        # use 20% of training data for validation
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = (len(dataset) - train_set_size)

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = data.random_split(
            dataset, [train_set_size, valid_set_size], generator=seed)

        # Create a data loader for batching your training data
        train_loader = DataLoader(
            train_set, batch_size=batch_size, num_workers=3, pin_memory=True)
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, num_workers=3, pin_memory=True)

        # Run learning rate finder
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader,
                                  val_dataloaders=valid_loader)

        # Results can be found in
        print(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        model.learning_rate = new_lr

        trainer.fit(model, train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
        trainer.save_checkpoint(
            datetime.now().isoformat()+criterion._get_name()+".ckpt")
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
