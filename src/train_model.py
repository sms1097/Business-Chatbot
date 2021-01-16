import json
import pickle
import numpy as np
import tensorflow as tf

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder


## Constants
EMBEDDING_DIM = 50 
EMBEDDINGS_PATH = '/home/sean/Documents/Business-Chatbot/data/glove.6B.50d.txt'
DATA_PATH = '/home/sean/Documents/Business-Chatbot/data/intents.json'


## Setup
patterns = []
tweeter = TweetTokenizer()
lemma = WordNetLemmatizer()

data = open(DATA_PATH).read()
data = json.loads(data)


# Flatten list
flatten = lambda l: np.array([x for sub in l for x in sub])
filter_words = lambda seq: [lemma.lemmatize(word) for word in seq if word not in stopwords.words('english')]


# Get input/targets in coupled lists
# also tokenize
for intent in data['intents']:
    patterns.append([
        (' '.join(tweeter.tokenize(pattern)), intent['tag']) 
        for pattern in intent['patterns']])


# flatten and move to numpy array
patterns = flatten(patterns)
patterns = np.apply_along_axis(filter_words, 0, patterns)

# setup for TF
X_tokenizer = Tokenizer(oov_token="<OOV>")
X_tokenizer.fit_on_texts(patterns[:,0])

X_train = X_tokenizer.texts_to_sequences(patterns[:,0])
X_train = pad_sequences(X_train, padding='post')

y_enc = LabelEncoder()
y_train = y_enc.fit_transform(patterns[:,1])
classes = len(y_enc.classes_)

print(y_enc.classes_)

vocab_size = len(X_tokenizer.word_index) + 1

# There is such limited training data that one batch is data set
data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
data = data.batch(X_train.shape[0])

## Initialize GloVe weights
embeddings_index = {}
with open(EMBEDDINGS_PATH) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in X_tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

## Training 
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, weights=[embeddings_matrix], mask_zero=True),
    tf.keras.layers.Dropout(0.4),  
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024)),
    tf.keras.layers.Dropout(0.4), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(classes, activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

history = model.fit(data, epochs=50)

## Save model
model.save_weights('/home/sean/Documents/Business-Chatbot/model/chatbot')

with open('/home/sean/Documents/Business-Chatbot/model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(X_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/sean/Documents/Business-Chatbot/model/label_encoder', 'wb') as handle:
    pickle.dump(y_enc, handle, protocol=pickle.HIGHEST_PROTOCOL)
