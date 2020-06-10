import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, email,re
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import json
import pickle
from sklearn.preprocessing import OneHotEncoder 
from keras.layers import *
from keras.models import Sequential


# from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from keras.preprocessing import image
# from keras.models import Model, load_model
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.layers import Input, Dense, Dropout, Embedding, LSTM
# from keras.layers.merge import add


from nltk.tokenize.regexp import RegexpTokenizer
import gensim
from gensim import corpora
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer


model = load_model("../best_model.h5")
model._make_predict_function()

def preprocess(string1, string2):
    s = string1 + string2
    return s


embeddings = {}
with open('../../data_science/Machine-learning-master/17. Word2Vec/glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        embeddings[word] = coeffs
    f.close()
print(len(embeddings))


def encode(s1, s2):
    string = preprocess(s1, s2)
#     feature_vector = model.predict(string)
#     feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector


session = tf.Session()
graph = tf.get_default_graph()
set_session(session)


with open("./storage/word_to_idx.pkl", 'rb') as w2i:
    word_to_idx = pickle.load(w2i)
    
with open("./storage/idx_to_word.pkl", 'rb') as i2w:
    idx_to_word = pickle.load(i2w)


def predict_class(string):
    
    return target_class


def caption_this_image(s1, s2):

    enc = encode(s1, s2)
    caption = predict_class(enc)
    
    return caption
