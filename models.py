import torch
from torch import nn
import pytorch_lightning as pl

class TextRNN(pl.LightningModule):
    def __init__(self, vocab_size, config):
        super(TextRNN, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers, 
                           dropout=config.dropout, batch_first=True)
        self.fc = nn.Linear(config.hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.reshape(-1, out.size(2)))
        return out, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.init_hidden(x.size(0))
        y_hat, _ = self(x, hidden)
        loss = self.loss(y_hat, y.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.config.num_layers, batch_size, self.config.hidden_size).zero_(),
                  weight.new(self.config.num_layers, batch_size, self.config.hidden_size).zero_())
        return hidden
