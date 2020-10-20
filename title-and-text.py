import nltk
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model

from keras.callbacks import ModelCheckpoint

# load the data
df_train = pd.read_csv("/content/drive/My Drive/fnd-usingML/dataset/cleaned/train.csv")
df_test = pd.read_csv("/content/drive/My Drive/fnd-usingML/dataset/cleaned/test.csv")
df_val = pd.read_csv("/content/drive/My Drive/fnd-usingML/dataset/cleaned/val.csv")

# Obtain additional stopwords from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Remove stopwords and remove words with 2 or less characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result

# Apply the function to the dataframe
df_train['clean'] = df_train['text'].apply(preprocess)
df_test['clean'] = df_test['text'].apply(preprocess)
df_val['clean'] = df_val['text'].apply(preprocess)

# Obtain the total words present in the dataset
list_of_words = []
for i in df_train.clean:
    for j in i:
        list_of_words.append(j)

# Obtain the total number of unique words
total_words = len(list(set(list_of_words)))

# join the words into a string
df_train['clean_joined'] = df_train['clean'].apply(lambda x: " ".join(x))
df_test['clean_joined'] = df_test['clean'].apply(lambda x: " ".join(x))
df_val['clean_joined'] = df_val['clean'].apply(lambda x: " ".join(x))

# length of maximum document will be needed to create word embeddings 
maxlen = -1
for doc in df_train.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)

x_train = df_train['clean_joined']
y_train = df_train['isfake']

x_test = df_test['clean_joined']
y_test = df_test['isfake']

x_val = df_val['clean_joined']
y_val = df_val['isfake']

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_val = np.asarray(y_val)

# Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words = total_words)

tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(x_test)
tokenizer.fit_on_texts(x_val)


train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)
val_sequences = tokenizer.texts_to_sequences(x_val)

# Add padding can either be maxlen = 4406 or smaller number maxlen = 40 seems to work well based on results
padded_train = pad_sequences(train_sequences,maxlen = maxlen, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = maxlen, truncating = 'post') 
padded_val = pad_sequences(val_sequences,maxlen = maxlen, truncating = 'post') 

# Sequential Model
model = Sequential()

# embeddidng layer
model.add(Embedding(total_words, output_dim = 128))
# model.add(Embedding(total_words, output_dim = 240))

# Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

filepath="./checkpoints/title-and-test/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, 
                             monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = 'max')

callbacks_list = [checkpoint]

# train the model
model.fit(padded_train, y_train, batch_size = 64, validation_data = (padded_val, y_val), epochs = 2, callbacks = callbacks_list)

# make prediction
pred = model.predict(padded_test)

# if the predicted value is >0.5 it is real else it is fake
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

# getting the accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)