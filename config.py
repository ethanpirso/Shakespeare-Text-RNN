class Config:
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.max_epochs = 1
        self.seq_length = 100
        self.embedding_dim = 256
        self.hidden_size = 512
        self.num_layers = 2
        self.dropout = 0.3
        self.generate_length = 1000
        self.data_path = 'shakespeare.txt'

config = Config()
