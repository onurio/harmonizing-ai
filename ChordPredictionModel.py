from lightning import lightning as L
import torch
from torch import nn


class ChordPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob):
        super(ChordPredictionModel, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim1, 256)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.dense2 = nn.Linear(256, output_dim)
        self.leakyRelu = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dense1(out)
        out = self.dropout3(out)
        out = self.dense2(out)
        # out = self.leakyRelu(out)
        return out


# device = torch.device("mps")


class ChordPredictionModelLightning(L.LightningModule):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob, learning_rate, criterion):
        super(ChordPredictionModelLightning, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim1, 256)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.dense2 = nn.Linear(256, output_dim)
        # self.leakyRelu = nn.Sigmoid()
        # self.signmoid = nn.Sigmoid()
        self.save_hyperparameters(ignore=['criterion'])

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dense1(out)
        out = self.dropout3(out)
        out = self.dense2(out)
        # out = self.leakyRelu(out)
        # out = self.signmoid(out)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss


class ChordPredictionModelLightningEmbedding(L.LightningModule):
    def __init__(self, note_vocab_size, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob, learning_rate, criterion):
        super(ChordPredictionModelLightningEmbedding, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.embedding = nn.Embedding(note_vocab_size, input_dim, 0)
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim1, 256)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.dense2 = nn.Linear(256, output_dim)
        self.long = torch.LongTensor
        self.float = torch.FloatTensor

        # self.leakyRelu = nn.Sigmoid()
        # self.signmoid = nn.Sigmoid()
        self.save_hyperparameters(ignore=['criterion'])

    def forward(self, x):
        out = self.embedding(x)
        out = self.float(out)
        out = self.lstm1(out)
        out = self.dropout1(out)
        out = self.lstm2(out)
        out = self.dropout2(out)
        out = self.lstm3(out)
        out = self.dense1(out)
        out = self.dropout3(out)
        out = self.dense2(out)
        out = self.long(out)
        # out = self.leakyRelu(out)
        # out = self.signmoid(out)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss
