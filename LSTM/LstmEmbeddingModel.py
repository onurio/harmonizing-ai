import torch.nn as nn


class MIDIChordModel(nn.Module):
    def __init__(self, num_unique_notes, embedding_dim, hidden_dim1, hidden_dim2, dropout_prob):
        super(MIDIChordModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_unique_notes, embedding_dim)

        # LSTM layers
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim1, batch_first=True)

        # Linear (dense) layers
        self.dense1 = nn.Linear(hidden_dim1, 256)
        # Adjust the output size
        self.dense2 = nn.Linear(256, num_unique_notes)

        # Dropout layers (optional)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)

    def forward(self, input_notes):
        # Embedding layer
        note_embedded = self.embedding(input_notes)

        # LSTM layers
        out, _ = self.lstm1(note_embedded)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)

        # Linear (dense) layers
        out = self.dense1(out)
        out = self.dropout3(out)
        out = self.dense2(out)  # Predict note indices

        # Activation function (optional)
        # out = self.leakyRelu(out)

        return out
