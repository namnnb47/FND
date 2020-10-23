import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from keras.models import load_model

model = load_model('./weights-improvement-01-0.68.hdf5')

app = Flask(__name__)

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer

num_words = 185729
maxlen = 0
tokenizer = Tokenizer(num_words = num_words)

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        data = request.args.get('text')
        pred = get_prediction(model, data)
        result = ''
        
        if pred[[0]] > 0.5:
            result = 'fake'
        else:
            result = 'true'

        print(pred[[0]])
        return {'result is ': result
                'weigh is ': round((pred*100),0) }

@app.route('/predict',methods = ['POST'])
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
    tokenizer.fit_on_texts(corpus)

    len_data = max([len(x)])
    if len_data<10:
        maxlen = 41
    elif len_data<100:
        maxlen = 231
    elif maxlen <1000:
        maxlen = 4927
    else:
        maxlen = 13682
    data_sequence = tokenizer.texts_to_sequences(corpus)
    padded_data = pad_sequences(data_sequence, maxlen=maxlen, truncating='post')
    return model.predict(padded_data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=8080)