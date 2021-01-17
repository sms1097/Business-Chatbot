import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

class ChatBot:
    def __init__(self):
        with open('model/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open('model/label_encoder', 'rb') as handle:
            self.y_enc = pickle.load(handle)

        self.tweeter = TweetTokenizer()
        self.lemma = WordNetLemmatizer()
        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 50, mask_zero=True),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                len(self.y_enc.classes_), activation='softmax')
        ])

        self.model.load_weights('model/chatbot')

        self.responses = self._load_responses()

    def _load_responses(self):
        flatten = lambda l: np.array([x for sub in l for x in sub])
        data = open('data/intents.json').read()
        data = json.loads(data)
        responses = []
        
        for intent in data['intents']:
            responses.append([
                (' '.join(self.tweeter.tokenize(response)), intent['tag']) 
                for response in intent['responses']])

        responses = flatten(responses)
        responses[:,1] = self.y_enc.transform(responses[:,1])
        return responses

    def _process_seq(self, seq):
        out_seq = [self.lemma.lemmatize(word) for word in seq if word not in stopwords.words('english')]
        out_seq = self.tokenizer.texts_to_sequences(out_seq)
        out_seq = pad_sequences(out_seq, padding='post')
        out_seq = out_seq[0]
        return out_seq

    def predict(self, seq):
        out_seq = self._process_seq(seq)
        label = self.model.predict(out_seq)
        label = np.argmax(label[-1,:])
        valid_responses = self.responses[np.where(self.responses[:,1] == str(label))][:,0]
        return np.random.choice(valid_responses)
