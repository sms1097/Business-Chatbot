# Business-Chatbot
This is a basic example for a chat bot that could serve as an interface between customers and a store. The specific 
example uses the conext of a gym, but this could easily be extended for other businesses. The data was limited, it's all data I wrote myself based upon gym FAQs in my area. Other than getting more data there are a few areas of improvement I've been thinking about with chatbots. 

### Brief Overview of Model Approach
This model learns to identify topics and randomly returns a response that is appropriate for the topic identified. The topics can be view in the `intents.json` file in the data folder. The data was pre-processed by tokenizing and lemmatizing the data, then feeding through an RNN. The model trained then predicts the topic and returns a resonse. 

### To Use
The tokenized models are too big to store in Github, so a docker image is available for reproducible results.
```
docker pull sms1097/gymbot:latest
docker run -p 8888:8888 gymbot
```
This will launch a jupyter notebook where the demo can be replicated. The model could also be retrained from this container. 

### Future Ideas for Chat Bots
#### Plugin Models on Base Model
One approach that would work well for training bots for multiple stores would be training a base model that has seen many examples and training a smaller plugin model for each individual store. This approach would help deal with problems in lacking case specific data. 

#### Spatial Interpolation for Disentangled Style Vectors
[John et al. (2018)](https://arxiv.org/pdf/1808.04339.pdf) showed it was possible to learn separate content and style vectors in an intermediate space between and encoder and decoder network. One idea I've been considering since stumbling across this paper is training a spatial ineterpolation model (IDW or Kriging, but kriging would be substantially harder) to generate style vectors for an individual based upon location. This assumes different dialects in language based on location, and would have a chatbot that closer resembled how the individual interfacing with the bot would expect text. <br>
This is an idea I got from my project on Formality Transfer, check out the project for that [here!](https://github.com/sms1097/formality-transfer).
