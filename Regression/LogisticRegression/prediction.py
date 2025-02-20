import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np  
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")


def optimization(text):
    stop_words = set(stopwords.words('turkish'))
    punctuation = ['.',',','?','!','"','(',')','[',']','{','}','-',
                   '_','+','=','*','/','\\','|','@','#','$','%','^',
                   '&','*','~','`',';',':','<','>','...','..','....',
                   "0","1","2","3","4","5","6","7","8","9","\"","'"]
    stop_words.update(punctuation)
    body = text
    body = body.lower()
    body = re.sub(r'http\S+', '', body)
    body = re.sub("\[[^]]*\]", "", body)
    body = (" ").join([word for word in body.split() if word not in stop_words])
    body = "".join([char for char in body if char not in punctuation])
    return body





prediction_model = pickle.load(open('Regression/LogisticRegression/model', 'rb'))
tfIdf = pickle.load(open("Regression/LogisticRegression/vectorizer", 'rb'))

def process_message(user_input):
    try:
        # Kullanıcı girdisini optimize et
        processed_input = optimization(user_input)
        prediction_input_vec = tfIdf.transform([processed_input])

        # Modeli kullanarak tahmin yap
        output_model = prediction_model.predict(prediction_input_vec)
        
        prediction_result = f"prediction.py çıktısı: {output_model}"
        print(f"Prediction işlemi tamamlandı: {prediction_result}")
        return prediction_result
    except Exception as e:
        error_msg = f"Prediction hatası: {str(e)}"
        print(error_msg)
        return error_msg


# Eğer varsa, input isteyen kodu kaldırın
# if __name__ == '__main__': gibi bir blok varsa ve içinde input() kullanıyorsa, silin
