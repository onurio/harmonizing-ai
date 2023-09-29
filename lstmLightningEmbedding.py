
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from ChordPredictionModel import ChordPredictionModelLightningEmbedding
from lightning import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from datetime import datetime
import random
from notevocab import create_note_vocabulary, midi_to_pitch
from torch.nn.utils.rnn import pad_sequence

# Example usage:
note_vocab = create_note_vocabulary(num_octaves=10, start_octave=1)
note_vocab_size = (len(note_vocab))


class ChordDataset(Dataset):
    def __init__(self, input_file, target_file, subset_percentage=100, padding_idx=0):
        self.input_data = pd.read_csv(input_file).values.tolist()
        self.target_data = []
        self.padding_idx = padding_idx

        with open(target_file, 'r') as file:
            for line in file:
                # Split each line by commas and convert to integers
                notes = line.strip().split(',')
                self.target_data.append(notes)

        self.subset_size = int(len(self.input_data) * subset_percentage / 100)

    def __len__(self):
        return self.subset_size

    def __getitem__(self, index):
        # Variable-length list of MIDI note indices
        sequence_length = len(self.target_data[index])
        target_padded = self.target_data[index]
        for i in range(12-sequence_length):
            target_padded.append(self.padding_idx)

        input_padded = [int(self.input_data[index][0])]
        for i in range(11):
            input_padded.append(self.padding_idx)

        input_padded = torch.LongTensor(input_padded)
        target_padded = torch.LongTensor(target_padded)
        return input_padded, target_padded


# Train the model
if __name__ == '__main__':
    # Set hyperparameters
    input_dim = 12
    hidden_dim1 = 256
    hidden_dim2 = 512
    output_dim = 12
    learning_rate = 0.001  # Adjust the learning rate as desired
    dropout_prob = 0.2
    batch_size = 64
    num_epochs = 10

    # first_criterion = nn.BCEWithLogitsLoss()
    # second_criterion = nn.BCELoss()
    # third_criterion = nn.MSELoss()
    forth_criterion = nn.CrossEntropyLoss()

    criterions = [forth_criterion]

    for criterion in criterions:
        # Initialize the model
        model = ChordPredictionModelLightningEmbedding(
            note_vocab_size, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob, learning_rate, criterion)
        trainer = L.Trainer(accelerator='mps', devices=1,
                            max_epochs=num_epochs, profiler="simple", callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)])

        # tuner = Tuner(trainer)

        # Load input and target data from separate CSV files
        # Replace with the path to your input CSV file
        input_csv_file = 'input_chords.csv'
        # Replace with the path to your target CSV file
        target_csv_file = 'output_chords.csv'

        dataset = ChordDataset(input_csv_file, target_csv_file, 1)

        for i in range(5):  # Print the first 5 samples
            input_data, target = dataset[i]
            print("Sample", i + 1)
            print("Input:", input_data)
            print("Target:", target)

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

        # # Run learning rate finder
        # lr_finder = tuner.lr_find(model, train_dataloaders=train_loader,
        #                           val_dataloaders=valid_loader)

        # # Results can be found in
        # print(lr_finder.results)

        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()

        # # Pick point based on plot, or get suggestion
        # new_lr = lr_finder.suggestion()

        # model.learning_rate = new_lr

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
