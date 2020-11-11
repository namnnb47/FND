import pandas as pd
from flask import Flask, url_for, request
from keras.models import load_model
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences
model_object = load_model('./checkpoints/text-only/weights-improvement-01-0.88.hdf5')
model_fakeNews = load_model('./checkpoints/text-only/weights-improvement-01-0.88.hdf5')

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
        result = ''
        data = request.args.get('text')
        pred = get_prediction(model_object, model_fakeNews, data)  
        if len(data) < 50:
            result = '-1'
        elif pred[0][0] = 2:
            result = '2'
        elif pred[0][0] < 0.5
            result = str(pred)
        else:
            result = str(pred)
        return {'result': result}    
    

def get_prediction(model1, model2, data):
    corpus = []
    new = re.sub('[^a-zA-Z]', ' ', data)
    new = new.lower()
    new = new.split()
    lemmatizer = WordNetLemmatizer()
    sw = stopwords.words('english')
    new = [lemmatizer.lemmatize(word) for word in new if not word in set(sw)]
    new = ' '.join(new)
    corpus.append(new)
    
    tokenizer.fit_on_texts(corpus)
    data_sequence = tokenizer.texts_to_sequences(corpus)
    padded_data = pad_sequences(data_sequence, maxlen=13682, truncating='post')
    result = model1.predict(padded_data)
    if result[0][0] > 0.5:
        return 2
    else:
        return model2.predict(padded_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5000)
