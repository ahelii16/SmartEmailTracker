import os, sys, email, re
import tensorflow as tf

import funct
from funct import clean,getOutputEmbeddings

from keras.layers import *
from keras.models import Sequential

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder

import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns;sns.set_style('whitegrid')

# NLP
from nltk.tokenize.regexp import RegexpTokenizer
from subprocess import check_output

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer

# emails_df = pd.read_csv('./emails.csv', nrows=20000)
df = pd.read_csv('/home/siddharth/Desktop/Codes/SmartEmailTracker-Siddharth.1.1/emaildataset.csv')
l = df.shape
print(l)

for i in range(df.shape[0]):
    # merge subject and body strings
    df['Text_Data'] = df['Subject'] + " " + df['Body']

text_clean=[]


for i in range(df.shape[0]):
    text_clean.append(clean(df.loc[i]['Text_Data']))

df['Text_Data'] = text_clean

train, test = train_test_split(df, test_size=0.2)

z1 = train.values
z2 = test.values

# Create Training and Testing Set
X_train = z1[:, 7]
Y_train = z1[:, 6]

X_test = z2[:, 7]
Y_test = z2[:, 6]

enc = OneHotEncoder(handle_unknown='ignore')

Y_train = Y_train.reshape(-1, 1)
Y_train = enc.fit_transform(Y_train)

enc2 = OneHotEncoder(handle_unknown='ignore')
Y_test = Y_test.reshape(-1, 1)
Y_test = enc2.fit_transform(Y_test)

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

XT = X_train
Xt = X_test

embed_matrix_train = getOutputEmbeddings(XT)
embed_matrix_test = getOutputEmbeddings(Xt)

model = Sequential()
model.add(LSTM(64,input_shape=(50,50),return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(64,input_shape=(50,50), return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(8))
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc']) # try optimizer='rmsprop'
#model.summary()

checkpt = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=True, save_best_only=True)
earlystop = EarlyStopping(monitor='val_acc', patience=10)


hist = model.fit(embed_matrix_train,Y_train,batch_size=32,epochs=40,shuffle=True,validation_split=0.2, callbacks=[checkpt, earlystop])

pred = model.predict_classes(embed_matrix_test)

print(model.evaluate(embed_matrix_test,Y_test))

