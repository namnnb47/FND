import pandas as pd
from flask import Flask, url_for, request
from keras.models import load_model
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences
prd_model = load_model('./checkpoints/text-only/weights-improvement-01-0.89.hdf5')
app = Flask(__name__)
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = 217500)


@app.route('/predict', methods=['POST'])
def sentiment():
    if request.method == 'POST':
        data = request.args.get('text')
        pred = get_prediction(prd_model, data)
        
        result = ''
        if pred[[0]] > 0.5:
            result = 'fake'
        else:
            result = 'true'
        return {'result': result}


def get_prediction(model, data):
    corpus = []
    new = re.sub('[^a-zA-Z]', ' ', data)
    new = new.lower()
    new = new.split()
    lemmatizer = WordNetLemmatizer()
    sw = stopwords.words('english')
    new = [lemmatizer.lemmatize(word) for word in new if not word in set(sw)]
    new = ' '.join(new)
    corpus.append(new)

    data_sequence = tokenizer.texts_to_sequences(corpus)
    padded_data = pad_sequences(data_sequence, maxlen=13682, truncating='post')
    return model.predict(padded_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5000)
