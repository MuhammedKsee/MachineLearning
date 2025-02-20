from flask import Flask, request, jsonify, send_file
import sys
import os
import traceback
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from googletrans import Translator

cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

logging.getLogger('werkzeug').setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_name = "microsoft/DialoGPT-medium" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

translator = Translator()

# Flask Uygulaması
app = Flask(__name__)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("\n=== Yeni İstek ===")
        print("Gelen istek:", request.data.decode('utf-8'))
        
        if not request.is_json:
            print("Hata: JSON verisi değil!")
            return jsonify({'response': 'JSON verisi bekleniyor'}), 400
            
        data = request.get_json()
        print("Alınan veri:", data)
        
        user_message = data.get('message', '')
        print(f"Kullanıcı mesajı: {user_message}")
        
        if not user_message:
            return jsonify({'response': 'Boş mesaj gönderilemez'}), 400
        
        translated_message = translator.translate(user_message, src='tr', dest='en').text
        print(f"İngilizceye çevrilen mesaj: {translated_message}")

        new_user_input_ids = tokenizer.encode(translated_message + tokenizer.eos_token, return_tensors='pt')

        chat_history = new_user_input_ids

        bot_input_ids = chat_history
        chat_history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        bot_output = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Modelin cevabı: {bot_output}")

        final_response = translator.translate(bot_output, src='en', dest='tr').text
        print(f"Türkçeye çevrilen yanıt: {final_response}")
        
        return jsonify({'response': final_response})
        
    except Exception as e:
        error_msg = f"Hata oluştu: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'response': error_msg})

if __name__ == '__main__':
    print("Sunucu başlatılıyor...")
    if app.debug:
        app.run(debug=True, port=5501, host='127.0.0.1')
    else:
        app.run(port=5501)
