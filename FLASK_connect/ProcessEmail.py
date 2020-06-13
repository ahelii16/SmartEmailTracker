# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:28:19 2020

@author: divya
"""

import numpy as np
import pandas as pd
import re

# NLP
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer

from keras.layers import *
from keras.models import Sequential
from keras.models import model_from_json



def clean(text):
    stop = set(stopwords.words('english'))
    stop.update(("to","cc","subject","http","from", "gbp", "usd", "eur", "cad", "sent","thanks", "acc", "ID", "account", "regards", "hi", "hello", "thank you"))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    porter= PorterStemmer()
    
    text=text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stem = " ".join(porter.stem(token) for token in normalized.split())
    
    return normalized

embeddings = {}
with open('/home/siddharth/Desktop/Codes/DataScience/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')

        embeddings[word] = coeffs
    f.close()
print(len(embeddings))

def getOutputEmbeddings(X):  
    X = X.split()
    embedding_matrix_output = np.zeros((1,50,50))
    for jx in range(min(100, len(X))):
        embedding_matrix_output[0][jx] = embeddings[X[jx].lower()]
            
    return embedding_matrix_output

#dependent on model loaded - replace with new classes
classes = ['BankFailed', 'BankProgress', 'BankComplete', 'BankRequest',
       'ClientProgress', 'ClientStatus', 'ClientComplete', 'ClientFailed']

def ProcessNewEmail(p_to, p_from, p_subject, p_body):
    test_str = p_subject + " " + p_body
    #clean email
    clean_test_str = clean(test_str)
    print(clean_test_str)
    emb_X = getOutputEmbeddings(clean_test_str)
    #load model - replace with new model
    with open("/home/siddharth/Python/EmailNew/model.json", "r") as file:
        model=model_from_json(file.read())
    model.load_weights("/home/siddharth/Python/EmailNew/best_model.h5")
    #model.summary()
    p = model.predict_classes(emb_X)
    #print (p.shape)
    return classes[p[0]]

p=ProcessNewEmail("abc@abc.com" , "xyz@xyz.com", "Payment failed", "Please look into the matter. \n thanks")
print(type(p),p)
