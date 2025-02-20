import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from model import TextDataset, TransformerLanguageModel, ModelTrainer, Config  # Modelinize ait dosyayı import edin
import re
import math

# Metin ön işleme ve temizleme fonksiyonu
class TextPreprocessor:
    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)  # Fazla boşlukları kaldır
        text = re.sub(r'[^\w\s]', '', text)  # Özel karakterleri kaldır
        return text.strip().lower()

# Tokenizer'ı yükleme ve kullanma
class CustomTokenizer:
    def __init__(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

# Modeli yükleme
def load_model(model_path, tokenizer_path):
    # Tokenizer ve model yükleme
    tokenizer = CustomTokenizer(tokenizer_path)
    model = TransformerLanguageModel(vocab_size=tokenizer.tokenizer.get_vocab_size())
    model.load_state_dict(torch.load(model_path))
    model = model.to(Config.DEVICE)  # Modeli doğru cihaza taşır
    model.eval()  # Modeli değerlendirme moduna al
    return model, tokenizer

# Model ile metin tahmini yapma
def generate_text(model, tokenizer, input_text, max_length=50):
    # Başlangıç metnini tokenize et
    tokens = tokenizer.encode(input_text)
    tokens_tensor = torch.tensor([tokens]).to(Config.DEVICE)
    
    # Modelin metni tamamlamasını başlat
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Modelden tahmin al
            outputs = model(tokens_tensor)
            
            # Sonuçları al ve en olası token'ı seç
            predicted_token = torch.argmax(outputs[0, -1, :], dim=-1).item()

            # Tahmin edilen token'ı tensöre dönüştür ve concatenate et
            predicted_token_tensor = torch.tensor([[predicted_token]]).to(Config.DEVICE)
            tokens_tensor = torch.cat([tokens_tensor, predicted_token_tensor], dim=1)
            
            # Yeni token ile devam et
            if predicted_token == tokenizer.encode('[SEP]')[0]:
                break  # Eğer SEP token'ına ulaşılırsa durdur
    
    # Sonuçları çözümle
    generated_text = tokenizer.decode(tokens_tensor[0].tolist())
    return generated_text


# Kullanım
def main():
    # Model ve tokenizer dosyalarını yükle
    model_path = "language_model.pth"
    tokenizer_path = "tokenizer.json"
    
    model, tokenizer = load_model(model_path, tokenizer_path)
    while True:

    # Kullanıcıdan input al
        input_text = input("Lütfen bir metin girin: ")
        input_text = TextPreprocessor.clean_text(input_text)  # Temizleme

    # Modeli kullanarak metin oluştur
        output_text = generate_text(model, tokenizer, input_text, max_length=50)
        print(f"Modelin ürettiği metin: {output_text}")

if __name__ == "__main__":
        main()
