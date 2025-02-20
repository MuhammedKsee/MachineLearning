import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
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
    # Örnek veri için daha küçük bir subset
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
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.trainer = trainers.BpeTrainer(special_tokens=Config.SPECIAL_TOKENS)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    def train(self, texts):
        self.tokenizer.train_from_iterator(texts, self.trainer)
    
    def save(self, path):
        self.tokenizer.save(path)
    
    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)

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

def main():
    # Load and preprocess data with trust_remote_code=True
    try:
        dataset = load_dataset(
            "wikipedia", 
            "20220301.en", 
            split="train", 
            trust_remote_code=True
        )
        # Veri setinden sadece belirli sayıda örnek al
        dataset = dataset.select(range(min(len(dataset), Config.NUM_SAMPLES)))
        texts = [entry['text'] for entry in dataset]
    except Exception as e:
        print(f"Wikipedia veri seti yüklenirken hata: {e}")
        # Hata durumunda örnek veri kullan
        texts = [
            "This is a sample text for training.",
            "We are using this as an example.",
            "Machine learning is fascinating.",
            # ... daha fazla örnek metin ekleyebilirsiniz
        ]

    cleaned_texts = [TextPreprocessor.clean_text(text) for text in texts]

    # Initialize and train tokenizer
    tokenizer = CustomTokenizer()
    tokenizer.train(cleaned_texts)
    tokenizer.save("tokenizer.json")

    # Create dataset and dataloader
    dataset = TextDataset(cleaned_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Initialize model and trainer
    model = TransformerLanguageModel(vocab_size=tokenizer.get_vocab_size())
    trainer = ModelTrainer(model, tokenizer)

    # Training loop
    print("Eğitim başlıyor...")
    for epoch in range(Config.EPOCH):
        loss = trainer.train(dataloader)
        print(f"Epoch {epoch+1}/{Config.EPOCH}, Loss: {loss:.4f}")

    # Save model
    trainer.save_model("language_model.pth")

    # Evaluation
    test_texts = ["This is a test sentence.", "Another test for the model."]
    test_dataset = TextDataset(test_texts, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    perplexity = trainer.calculate_perplexity(test_dataloader)
    print(f"Model Perplexity: {perplexity:.2f}")

    # Test generation
    test_text = "The quick brown fox"
    output = trainer.evaluate(test_text)
    print(f"Input: {test_text}")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()