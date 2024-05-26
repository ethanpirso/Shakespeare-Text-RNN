import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import os
from utils import preprocess_text, create_vocab, text_to_sequences

class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        x = self.text[idx:idx+self.seq_length]
        y = self.text[idx+1:idx+self.seq_length+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class TextDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"{self.config.data_path} does not exist.")
        
        with open(self.config.data_path, 'r') as file:
            text = file.read()
        
        text = preprocess_text(text)
        self.chars, self.char2idx, self.idx2char = create_vocab(text)
        self.encoded_text = text_to_sequences(text, self.char2idx)

    def setup(self, stage=None):
        dataset = TextDataset(self.encoded_text, self.config.seq_length)
        self.train_set, self.val_set = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.config.batch_size)
