import json
import numpy as np

from nltk.tokenize import TweetTokenizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


## Setup
patterns = []
responses = []
tweeter = TweetTokenizer()

data = open("intents.json").read()
data = json.loads(data)


# Flatten list
flatten = lambda l: [x for sub in l for x in sub]

# Get input/targets in coupled lists
# also tokenize
for intent in data['intents']:
    patterns.append([(' '.join(tweeter.tokenize(pattern)), intent['tag']) for pattern in intent['patterns']])
    responses.append([(' '.join(tweeter.tokenize(response)), intent['tag']) for response in intent['responses']])


# flatten and move to numpy array
patterns, responses = flatten(patterns), flatten(responses)
patterns, responses = np.array(patterns), np.array(responses)


# setup for TF
X_tokenizer = Tokenizer(oov_token="<OOV>")
X_tokenizer.fit_on_texts(patterns[:,0])

X_train = X_tokenizer.texts_to_sequences(patterns[:,0])
X_train = pad_sequences(X_train, padding='post')

y_train = patterns[:,1]

print(y_train)

## Training 

