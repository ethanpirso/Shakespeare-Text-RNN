# Text Generation with RNN - Shakespeare Style

This project aims to implement a Recurrent Neural Network (RNN) to generate text that mimics the style of Shakespeare's writings. We utilize PyTorch Lightning to streamline our implementation, focusing on data preprocessing, model building, training, and text generation.

## Directory Structure
```
text_generation_project/
│
├── main.py          # Main training and generation script
├── models.py        # Model definition
├── datamodule.py    # Data preparation
├── config.py        # Configuration
├── utils.py         # Utility functions for text processing
└── requirements.txt # Python dependencies
```

## Steps

### 1. Data Preprocessing
Proper data preprocessing is essential to transform raw text data into a format suitable for RNNs. We follow these steps:

1. **Read and preprocess the text**: The text is converted to lowercase, and punctuation is removed to focus on the core text data.
2. **Create vocabulary**: We create a vocabulary of unique characters from the text and map each character to a unique index.
3. **Convert text to sequences**: The text is converted into sequences of integers based on the character mappings, making it suitable for the RNN input.

#### Preprocessing Code:
```python
# utils.py

import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def create_vocab(text):
    chars = sorted(set(text))
    char2idx = {ch: idx for idx, ch in enumerate(chars)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    return chars, char2idx, idx2char

def text_to_sequences(text, char2idx):
    return [char2idx[ch] for ch in text]
```

### 2. Model Implementation
We successfully implement an RNN using PyTorch Lightning, with the option to use LSTM layers for better performance on sequence data. The architecture is defined as follows:

- **Embedding layer**: Converts input indices to dense vectors of a fixed size.
- **LSTM layers**: Two LSTM layers process the sequences, capturing temporal dependencies.
- **Fully connected layer**: Maps the LSTM output to the vocabulary size for character prediction.

#### Model Code:
```python
# models.py

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
```

### 3. Training Process
The model is trained on sequences of characters from Shakespeare's writings. Training involves:

- **Data preparation**: Splitting the data into training and validation sets.
- **Training loop**: Training the model for a set number of epochs, periodically saving the model weights based on validation performance.

#### Training Code:
```python
# main.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from models import TextRNN
from datamodule import TextDataModule
from config import config
from utils import create_vocab

def main():
    data_module = TextDataModule(config)
    data_module.prepare_data()
    
    vocab_size = len(data_module.chars)
    model = TextRNN(vocab_size, config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename='rnn-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
```

### 4. Text Generation
Using the trained model, we generate new text that mimics the style of Shakespeare's writings. The process involves:

- **Starting with a seed text**: Initial text input to start the generation.
- **Generating characters sequentially**: Using the model to predict the next character based on the previous characters.

#### Generation Code:
```python
# main.py (continued)

def generate_text(model, config, char2idx, idx2char, start_text, length=1000):
    model.eval()
    chars = [char2idx[ch] for ch in start_text]
    generated_text = start_text

    hidden = model.init_hidden(1)
    for _ in range(length):
        x = torch.tensor([chars[-config.seq_length:]], dtype=torch.long)
        output, hidden = model(x, hidden)
        prob = torch.nn.functional.softmax(output[-1], dim=0).detach().cpu().numpy()
        next_char = np.random.choice(len(prob), p=prob)
        chars.append(next_char)
        generated_text += idx2char[next_char]

    return generated_text

if __name__ == '__main__':
    main()
    start_text = "To be or not to be "
    generated_text = generate_text(model, config, data_module.char2idx, data_module.idx2char, start_text, config.generate_length)
    print(generated_text)
```

### Success Criteria
- **Data Preprocessing**: The text data is preprocessed and converted into sequences suitable for RNNs.
- **Model Implementation**: A successful RNN implementation using PyTorch Lightning.
- **Text Generation**: The generated text should show coherent structure and stylistic similarities with the original Shakespearean text.
- **Documentation**: Clear documentation of architecture decisions, training process, and generation method.

### Requirements
List the necessary dependencies in `requirements.txt`:
```
torch
torchvision
pytorch-lightning
```

### Installation
To install the dependencies, run:
```
pip install -r requirements.txt
```

### Running the Project
To train the model and generate text, run:
```
python main.py
```

### Conclusion
This project demonstrates the implementation of an RNN for text generation, focusing on preprocessing, model training, and generating Shakespearean-style text. The modular approach ensures maintainability and scalability.

## Authors
Ethan Pirso