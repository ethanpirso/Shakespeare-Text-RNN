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
