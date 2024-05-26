import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from models import TextRNN
from datamodule import TextDataModule
from config import config
import torch
import numpy as np

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

    start_text = "To be or not to be "
    generated_text = generate_text(model, config, data_module.char2idx, data_module.idx2char, start_text, config.generate_length)
    print(generated_text)

if __name__ == '__main__':
    main()
