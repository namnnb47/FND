import pandas as pd
from flask import Flask, url_for, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
import h5py
import tensorflow
import sklearn
from tensorflow.keras.preprocessing.sequence import pad_sequences
#classify_model = load('./SVM_model')
#classify_model = load_model('./weight-improvement-detectAll')
fakeNews_model = load_model('./weights-improvement-03-0.88.hdf5')
classify_model = load_model('./weights-improvement-03-0.86.hdf5')
app = Flask(name)
CORS(app)
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer_fnd = Tokenizer(num_words = 185729)
tokenizer_clf = Tokenizer(num_words = 390061)
data_length_fnd = 275
data_length_clf = 185

@app.route('/predict', methods=['POST'])
def sentiment():
    if request.method == 'POST':
        result = ''
        data = request.json.get('text')
        print(type(data))
        #import ipdb;ipdb.set_trace() 
        classify_pred = get_classify__prediction(classify_model, data)
        lendata = get_len(data)
        if lendata < 20:
             result = -1
        elif classify_pred[0][0] > 0.5:
             result = classify_pred[0][0]+1
        else:
             fn_pred = get_fake_new_prediction(fakeNews_model, data)
             result = fn_pred[0][0]
        return  str(result)
        #return str(oj_pred)
def get_fake_new_prediction(model,  data):
    corpus = []
    new = re.sub('[^a-zA-Z]', ' ', data)
    new = new.lower()
    new = new.split()
    lemmatizer = WordNetLemmatizer() 
    sw = stopwords.words('english')
    new = [lemmatizer.lemmatize(word) for word in new if not word in set(sw)]
    new = ' '.join(new)
    corpus.append(new)
    tokenizer_fnd.fit_on_texts(corpus)
    data_sequence = tokenizer_fnd.texts_to_sequences(corpus)
    padded_data = pad_sequences(data_sequence, maxlen=data_length_fnd)
    result =  model.predict(padded_data)
    print('fakenew result is: ', result)
    return result
   
def get_classify__prediction(model, data):
    corpus = []
    new = re.sub('[^a-zA-Z]', ' ', data)
    new = new.lower()
    new = new.split()
    lemmatizer = WordNetLemmatizer() 
    sw = stopwords.words('english')
    new = [lemmatizer.lemmatize(word) for word in new if not word in set(sw)]
    new = ' '.join(new)
    corpus.append(new)
    tokenizer_clf.fit_on_texts(corpus)
data_sequence = tokenizer_clf.texts_to_sequences(corpus)
    padded_data = pad_sequences(data_sequence, maxlen=data_length_clf)
    result =  model.predict(padded_data)
    print('classify result is: ', result)
    return result

def get_len(data):
    text=data.split()
    return len(text)


if name == "main":
    app.run(host="0.0.0.0", debug=False, port=8080, ssl_context=('cert.pem', 'key.pem'))