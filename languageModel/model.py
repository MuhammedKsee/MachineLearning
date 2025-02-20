import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
import re
import math

class Config:
    """Model and training configuration"""
    EPOCH = 50
    BATCH_SIZE = 32
    MAX_LENGTH = 128
    D_MODEL = 512
    N_HEADS = 8
    N_LAYERS = 6
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    NUM_SAMPLES = 1000  

class TextPreprocessor:
    """Text cleaning and preprocessing utilities"""
    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text.strip().lower()

class CustomTokenizer:
    """Tokenizer wrapper class"""
    def __init__(self, path=None):
        if path:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(path)
        else:
            self.tokenizer = None

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def train(self, texts):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(special_tokens=Config.SPECIAL_TOKENS)
        tokenizer.train_from_iterator(texts, trainer)
        self.tokenizer = tokenizer

    def save(self, path):
        self.tokenizer.save(path)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

class TextDataset(Dataset):
    """Custom dataset for text data"""
    def __init__(self, texts, tokenizer, max_length=Config.MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])
        tokens = tokens[:self.max_length] + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens)

class TransformerLanguageModel(nn.Module):
    """Transformer-based language model"""
    def __init__(self, vocab_size, d_model=Config.D_MODEL, 
                 n_heads=Config.N_HEADS, n_layers=Config.N_LAYERS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, n_heads, n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, model, tokenizer, device=Config.DEVICE):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def calculate_perplexity(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), batch.view(-1))
                total_loss += loss.item() * batch.size(0)
                total_tokens += batch.size(0) * batch.size(1)

        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)

    def evaluate(self, text, max_length=50):
        self.model.eval()
        tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([tokens]).to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predicted_token = torch.argmax(outputs[0], dim=-1)
        return self.tokenizer.decode(predicted_token.tolist())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
