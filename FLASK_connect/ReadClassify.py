#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, email, re
import numpy as np
import pickle
import h5py as h5
import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer

from keras.models import Sequential
from keras.models import model_from_json

# In[14]:


from datetime import date

from msgbody import subjectGenerator, bodyGenerator


def clean(text):
    stop = set(stopwords.words('english'))
    stop.update(("to", "cc", "subject", "http", "from", "gbp", "usd", "eur", "cad", "sent", "thanks", "acc", "id",
                 "account", "regards", "hi", "hello", "thank you"))
    operators = set(('and', 'or', 'not'))
    stop = set(stop) - operators
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    porter = PorterStemmer()

    text = text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if ((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stem = " ".join(porter.stem(token) for token in normalized.split())

    #     return normalized, amount
    return normalized


# In[5]:


embeddings = {}
with open('/home/siddharth/Desktop/Codes/DataScience/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')

        #         print(word)
        #         print(coeffs)
        embeddings[word] = coeffs
    f.close()
print(len(embeddings))


# In[6]:


def getOutputEmbeddings(X):
    # let 10 is max len of sentence, 50 batch size (no. of e.g.)
    embedding_matrix_output = np.zeros((X.shape[0], 50, 50))

    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            # go to every word in current(ix) sentence
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]

    return embedding_matrix_output


# In[7]:


with open('/home/siddharth/Python/EmailNew/model_pickle', 'rb') as f:
    model = pickle.load(f)


with open("/home/siddharth/Python/EmailNew/model.json", "r") as file:
    model = model_from_json(file.read())
model.load_weights("/home/siddharth/Python/EmailNew/best_model.h5")
# In[30]:


text_body = ''


def Preprocess(fr, to, sub, body):
    text_body = sub + ' ' + body
    x, s = predictFunc()
    res = int(x)
    #k = replyFunc(res, s)
    return res


# In[31]:


def findNum(st):
    a = []
    for word in st.split():
        try:
            a.append(float(word))
        except ValueError:
            pass
    return a


# In[32]:


def predictFunc():
    clean_text = []
    s = findNum(text_body)
    clean_text.append(clean(text_body))
    df = pd.DataFrame(clean_text, columns=['test'])
    z1 = df.values
    x_t = z1[:, 0]
    embed_matrix_test = getOutputEmbeddings(x_t)
    pred = model.predict_classes(embed_matrix_test)
    return pred[0], str(s)


n = Preprocess('a', 'b', 'transaction failed for ID: 131231', 'The transaction is Failed. Thanks you!')
print(n)
