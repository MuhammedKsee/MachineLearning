from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Hugging Face model ve tokenizer'ı yükleme
MODEL_NAME = "bigcode/santacoder"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def generate_code(prompt):
    """
    Hugging Face modelini kullanarak kod üretir.
    """
    try:
        # Girdi metnini token'lara dönüştürme
        inputs = tokenizer(prompt, return_tensors="pt")

        # Modeli kullanarak kod oluşturma
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=150,  # Üretilecek kodun maksimum uzunluğu
            temperature=0.7, # Çeşitli ve tutarlı çıktı için
            num_beams=5,     # Beam search kullanımı
            early_stopping=True
        )
        
        # Üretilen kodu çözümleme
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code
    except Exception as e:
        return f"Hata: {str(e)}"

if __name__ == "__main__":
    print("Hugging Face Kod Üretici Konsol Uygulamasına Hoş Geldiniz!")
    print("Kod üretmek için bir istem girin ('çıkış' yazarak çıkabilirsiniz):")

    while True:
        prompt = input("\nİstem (Prompt): ")
        if prompt.lower() == "çıkış":
            print("Uygulama sonlandırıldı. Hoşça kalın!")
            break

        # Kod üretimi
        print("\nÜretilen Kod:")
        print(generate_code(prompt))
