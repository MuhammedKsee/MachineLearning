from flask import Flask, request, jsonify, send_file, send_from_directory
import sys
import os
import traceback
import json
import logging

# Flask'ın development server uyarısını gizle
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

# Log seviyesini ayarla
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Regression.LogisticRegression.prediction import process_message

app = Flask(__name__, static_folder='static')

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
        
        # Prediction işlemi
        response = process_message(user_message)
        print(f"Model yanıtı: {response}")
        
        return jsonify({'response': response})
        
    except Exception as e:
        error_msg = f"Hata oluştu: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'response': error_msg})

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("Sunucu başlatılıyor...")
    if app.debug:
        app.run(debug=True, port=5500, host='127.0.0.1')
    else:
        app.run(port=5500) 