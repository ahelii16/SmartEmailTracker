#loads biLSTM model
import re
import tensorflow as tf
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns;

sns.set_style('whitegrid')

# NLP
from nltk.tokenize.regexp import RegexpTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# import gensim
import spacy
import en_core_web_sm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

# emails_df = pd.read_csv('./emails.csv', nrows=20000)

#read from emaildatasetnew.csv? but that doesn't have textdata combined
#uncomment commented code
df = pd.read_csv('./emaildataset.csv')

nlp_ = en_core_web_sm.load()

nlp = spacy.load('en')


def clean(text):
    text = text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = " ".join([i for i in text.lower().split()])
    print(text)

    customize_stop_words = ["cc", "subject", "http", "gbp", "usd", "eur", "inr", "cad", "thanks", "acc", "id",
                            "account", "regards", "hi", "hello", "thank you", "greetings"]
    rem_stop = ["not", "wasn't", "hadn't", "won't", "can't", "didn't"]

    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True

    for w in rem_stop:
        nlp.vocab[w].is_stop = False

    doc = nlp(text)

    normalized = " ".join(token.lemma_ for token in doc if not token.is_stop)

    doc = " ".join(token.orth_ for token in nlp(normalized) if not token.is_punct | token.is_space)
    return doc


#for i in range(df.shape[0]):
    # merge subject and body strings
#    df['Text_Data'] = (df['Subject'] + " " + df['Body'])


#searches for transaction id in subject, returns '' if not found
def findNum(st):
    a = []
    for word in st.split():
        try:
            a.append(int(word))
        except ValueError:
            pass

    if (len(a)>0):
        return str(a[0])
    return ''


def converter(x):
    try:
        return ' '.join([x.lower() for x in str(x).split()])
    except AttributeError:
        return None  # or some other value


#df['Text_Data'] = df['Text_Data'].apply(converter)

#text_clean = []

#for i in range(df.shape[0]):
#    text_clean.append(clean(df.loc[i]['Text_Data']))

#df['Text_Data'] = text_clean

#lengths = []

#for i in range(df.shape[0]):
#    words = df.values[i][7].split()
#    lengths.append(len(words))

#avg = int(np.mean(lengths) + 2 * np.std(lengths))

#will change when model is retrained
avg=21

embeddings_index = {}
with open('./glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')

        #         print(word)
        #         print(coeffs)
        embeddings_index[word] = coeffs
    f.close()

t = Tokenizer()
l = list(df.Text_Data)
t.fit_on_texts(l)
vocab_size = len(t.word_index) + 1
# print(vocab_size)

# integer encode the mails
encoded_mails = t.texts_to_sequences(l)

# post padding
padded_inputs = pad_sequences(encoded_mails, maxlen=avg, padding='post')

embedding_matrix_train = np.zeros((vocab_size, 300))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_train[i] = embedding_vector

le = OneHotEncoder()
classes = df['Class'].values

classes = classes.reshape(-1, 1)
Y = le.fit_transform(classes)

#returns transaction id and predicted class
def inputfunc(to,fr,input_subj,input_bod):
    ID = findNum(input_subj)
    #print(ID)
    #print(type(ID))
    s1 = converter(input_bod + " " + input_subj)
    s = clean(s1)
    encoded_mail = t.texts_to_sequences([s])
    #print(encoded_mail)

    # post padding
    padded_input = pad_sequences(encoded_mail, maxlen=avg, padding='post')
    #padded_input.shape
    with open("model_glove.json", "r") as file:
        model_glove = model_from_json(file.read())
    model_glove.load_weights("model_glove.h5")
    y_pred = model_glove.predict(padded_input)
    k = le.inverse_transform(y_pred)
    return k[0][0] ,ID

#inr=inputfunc('a','b',"Transaction failed for ID: 3437386475674385","Hi, your payment with ID: 3437386475674385 for 98, 453 GBP was failed and amount will be transferred back in 3-5 business days.")
#print(inr)
#print(type(inr))
#print(inr[0][0])
#print(type(inr[0][0]))
