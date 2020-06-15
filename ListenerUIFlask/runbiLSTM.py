import re
import numpy as np
import pandas as pd


from nltk.tokenize.regexp import RegexpTokenizer
import spacy
import en_core_web_sm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import model_from_json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./emaildataset.csv')

nlp_ = en_core_web_sm.load()

nlp = spacy.load('en')

def clean(text):
    text = text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = " ".join([i for i in text.lower().split()])

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


def converter(x):
    try:
        return ' '.join([x.lower() for x in str(x).split()])
    except AttributeError:
        return None  # or some other value


avg=21

embeddings_index = {}

with open('glove.6B.300d.txt', encoding='utf-8') as f:
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


def inputfunc(to,fr,input_subj,input_bod):
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
    return k[0][0]

#inr=inputfunc('a','b',"Transaction failed for ID: 3437386475674385","Hi, your payment with ID: 3437386475674385 for 98, 453 GBP was failed and amount will be transferred back in 3-5 business days.")
#print(f'======>>{inr}')
#print(type(inr))
#print(inr[0][0])
#print(type(inr[0][0]))
