import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from keras.models import load_model

app = Flask(__name__)

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = 217500)

@app.route('/predict',methods = ['POST'])
def predict():
	data = request.args.get('text')
	corpus = []
	temp = re.sub('[^a-zA-Z]', ' ', data)
	temp = temp.lower()
	temp = temp.split()
	lemmatizer = WordNetLemmatizer() 
	sw = stopwords.words('english')
	temp = [lemmatizer.lemmatize(word) for word in temp if not word in set(sw)]
	temp = ' '.join(temp)
	corpus.append(temp)

	text_sequences = tokenizer.texts_to_sequences(corpus)
	padded_text = pad_sequences(text_sequences,maxlen = 100, truncating = 'post')
	model = load_model('./weights-improvement-01-0.68.hdf5')
	prediction = model_load.predict(padded_text)
	result = ''
	# for i in range(len(prediction_text_test)):
	if prediction[0] > 0.5:
	    result = 'fake'
	else:
	    result = 'true'
	print(prediction[[0]])
	return {'result': result}

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=8080)