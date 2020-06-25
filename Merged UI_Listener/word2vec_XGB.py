import pandas as pd
import numpy as np
import joblib
from wordfile import func
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy import spatial
import spacy
import xgboost


embeddings_index = {}
with open('./glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coeffs
    f.close()

nlp = spacy.load('en')

my_stop = ["'d", "'ll", "'m", "'re", "'s", "'ve", 'a', 'cc', 'subject', 'http', 'gbp', 'usd', 'eur', 'inr', 'cad',
           'thanks', "acc", "id", 'account', 'regards', 'hi', 'hello', 'thank you', 'greetings', 'about', 'above',
           'across', 'after', 'afterwards', 'against', 'alone', 'along', 'already', 'also', 'although', 'am', 'among',
           'amongst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere',
           'are', 'around', 'as', 'at', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
           'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'both', 'bottom', 'but', 'by',
           'ca', 'call', 'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'due', 'during', 'each', 'eight',
           'either', 'eleven', 'else', 'elsewhere', 'every', 'everyone', 'everything', 'everywhere', 'fifteen', 'fifty',
           'first', 'five', 'for', 'former', 'formerly', 'forty', 'four', 'from', 'front', 'further', 'get', 'give',
           'go', 'had', 'has', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon',
           'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into',
           'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'made', 'make',
           'many', 'may', 'me', 'meanwhile', 'might', 'mine', 'more', 'moreover', 'mostly', 'move', 'much', 'must',
           'my', 'myself', 'name', 'namely', 'neither', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'now',
           'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise',
           'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'quite',
           'rather', 're', 'really', 'regarding', 'same', 'say', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious',
           'several', 'she', 'should', 'show', 'side', 'since', 'six', 'sixty', 'so', 'some', 'somehow', 'someone',
           'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'take', 'ten', 'than', 'that', 'the',
           'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
           'thereupon', 'these', 'they', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru',
           'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'under', 'unless',
           'until', 'up', 'upon', 'us', 'used', 'using', 'various', 'very', 'via', 'was', 'we', 'well', 'were',
           'whatever', 'whence', 'whenever', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever',
           'whether', 'which', 'while', 'whither', 'whoever', 'whole', 'whom', 'whose', 'will', 'with', 'within',
           'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', '‘d', '‘ll', '‘m', '‘re', '‘s', '‘ve',
           '’d', '’ll', '’m', '’re', '’s', '’ve']


def get_only_chars(text):
    text = text.replace("-", " ")
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")

    text = text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    t = ""

    for i in text.lower().split():
        if func(i) is not None:
            t += func(i) + " "
        else:
            t += i + " "

    t = t.rstrip()

    text = " ".join([i for i in t.lower().split()])
    text = " ".join(token for token in text.split() if token not in my_stop)

    doc = nlp(text)
    normalized = " ".join(token.lemma_ for token in doc)
    doc = " ".join(token.orth_ for token in nlp(normalized) if not token.is_punct | token.is_space)
    return doc


def transform_sentence(text, embeddings_index):
    def preprocess_text(raw_text, model=embeddings_index):
        raw_text = raw_text.split()
        return list(filter(lambda x: x in embeddings_index.keys(), raw_text))

    tokens = preprocess_text(text)

    if not tokens:
        return np.zeros(300)

    c = [embeddings_index[i] for i in tokens]
    text_vector = np.mean(c, axis=0)
    return np.array(text_vector)


le = joblib.load('./pkl_objects/labelencoder.pkl')
clf = joblib.load('./pkl_objects/clf.pkl')


def findNum(st):
    a = []
    k=''
    for word in st.split():
        try:
            a.append(float(word))
        except ValueError:
            pass
    if not a:
        k = '000000'
    else:
        k = int(a[0])
    return str(k)


def inp(emailto, emailfrom, subj, bod):
    text = subj + " " + bod
    ID = str(findNum(text))
    text = get_only_chars(text)
    X_test_mean = np.array([transform_sentence(text, embeddings_index)])

    y_pred = clf.predict(X_test_mean)
    k = le.inverse_transform(y_pred)
    return k[0], ID


#k, ID = inp("fvf", "defrfg", "payment processed 123456",
#            "hi, the payment for acc 1234 for usd 3456 was paid successfully.")
#print(k, type(k), ID, type(ID))