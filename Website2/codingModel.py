from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Hugging Face modeli ve tokenizer'ı yükleme
MODEL_NAME = "codeparrot/codeparrot-small"  # Model adını kontrol edin
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")

def generate_code(prompt):
    """
    Hugging Face modelini kullanarak kod üretme.
    """
    try:
        # Girdi metnini token'lara dönüştürme
        inputs = tokenizer(prompt, return_tensors="pt")

        # Modeli kullanarak kod oluşturma
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=150,  # Üretilecek kodun maksimum uzunluğu
            num_beams=5,     # Beam search kullanımı
            early_stopping=True
        )
        
        # Üretilen kodu çözümleme
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code
    except Exception as e:
        return f"Hata: {str(e)}"

@app.route('/generate', methods=['POST'])
def generate():
    """
    Sunucudan prompt alarak kod üreten uç nokta.
    """
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "Boş prompt gönderilemez!"}), 400

    generated_code = generate_code(prompt)
    
    # JavaScript tarafında beklenen formatta yanıt döndür
    return jsonify({"code": generated_code.splitlines()} )  # Her satırı ayrı bir eleman olarak döndür

if __name__ == '__main__':
    app.run(port=5501, host='127.0.0.1', debug=True)
