#!/usr/bin/env python
# coding: utf-8

# # Few-Shot Learning Email Classification with Pre-Trained Word2Vec Embeddings

import pandas as pd
import numpy as np
from random import seed
from random import sample
from wordfile import func
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import gensim.downloader as api
from gensim.models.keyedvectors import Word2VecKeyedVectors
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
from scipy import spatial
import os
import spacy


model = KeyedVectors.load_word2vec_format('/home/siddharth/Downloads/GoogleNews-vectors-negative300.bin', binary=True)


def train():
    df = pd.read_csv("./emaildataset.csv", usecols = ['Subject','Body', 'Class'])
    df.head()

    nlp = spacy.load('en')


    my_stop = ["'d", "'ll", "'m", "'re", "'s", "'ve",'a','cc','subject','http', 'gbp', 'usd', 'eur', 'inr', 'cad', 'thanks', "acc", "id", 'account', 'regards', 'hi', 'hello', 'thank you', 'greetings', 'about','above', 'across','after','afterwards','against','alone','along','already','also','although','am','among', 'amongst','amount','an','and','another','any','anyhow','anyone','anything','anyway','anywhere','are','around','as', 'at','be','became','because','become','becomes','becoming','been','before','beforehand','behind','being','below', 'beside','besides','between','both','bottom','but','by','ca','call','can','could','did', 'do', 'does', 'doing', 'down', 'due', 'during', 'each', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'every', 'everyone', 'everything', 'everywhere', 'fifteen', 'fifty', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'four', 'from', 'front', 'further', 'get', 'give', 'go', 'had', 'has', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'made', 'make', 'many', 'may', 'me', 'meanwhile', 'might', 'mine', 'more', 'moreover', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'quite', 'rather', 're', 'really', 'regarding', 'same', 'say', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'under', 'unless', 'until', 'up', 'upon', 'us', 'used', 'using', 'various', 'very', 'via', 'was', 'we', 'well', 'were', 'whatever', 'whence', 'whenever', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'whoever', 'whole', 'whom', 'whose', 'will', 'with', 'within', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', '‘d', '‘ll', '‘m', '‘re', '‘s', '‘ve', '’d', '’ll', '’m', '’re', '’s', '’ve']


    def get_only_chars(text):
        text = text.replace("-", " ") #replace hyphens with spaces
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")

        text=text.rstrip()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        t = ""

        for i in text.lower().split():
            if func(i) is not None:
                t += func(i) + " "
            else :
                t += i + " "

        t = t.rstrip()

        text = " ".join([i for i in t.lower().split()])
        text = " ".join(token for token in text.split() if token not in my_stop)

        doc = nlp(text)

        normalized = " ".join(token.lemma_ for token in doc)

        doc = " ".join(token.orth_ for token in nlp(normalized) if not token.is_punct | token.is_space)
        return doc



    for i in range(df.shape[0]):
        # merge subject and body strings
        df['Text'] = (df['Subject'] + " " + df['Body'])

    def converter(x):
        try:
            return ' '.join([x.lower() for x in str(x).split()])
        except AttributeError:
            return None  # or some other value

    df['Text'] = df['Text'].apply(converter)


    text_clean=[]

    for i in range(df.shape[0]):
        text_clean.append(get_only_chars(df.loc[i]['Text']))


    df['Text'] = df['Text'].apply(lambda x: get_only_chars(x))

    df = df.drop_duplicates('Text')


    df.sample(frac=1).reset_index(drop=True)



    # set the by default to:
    num_classes = df.Class.unique() # the number of classes we consider (since the dataset has many classes)
    sample_size = 2 # the number of labeled sampled we’ll require from the user



    smallest_sample_size = min(df['Class'].value_counts())


    # Generate samples that contains K samples of each class

    def gen_sample(sample_size, num_classes):

        df_1 = df[(df["Class"] < num_classes+1)].reset_index().drop(["index"], axis=1).reset_index().drop(["index"], axis=1)

        train = df_1[df_1["Class"] == np.unique(df_1['Class'])[0]].sample(sample_size)
        #     return train
        train_index = train.index.tolist()

        for i in range(1,num_classes):
            train_2 = df_1[df_1["Class"] == np.unique(df_1['Class'])[i]].sample(sample_size)
            train = pd.concat([train, train_2], axis=0)
            train_index.extend(train_2.index.tolist())

        test = df_1[~df_1.index.isin(train_index)]
        return train, test



    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])



    df['Class'] = df['Class'].apply(lambda x : x + 1)


    # Text processing (split, find token id, get embedidng)
    def transform_sentence(text, model):

        def preprocess_text(raw_text, model=model):

            raw_text = raw_text.split()
            return list(filter(lambda x: x in model.vocab, raw_text))

        tokens = preprocess_text(text)

        if not tokens:
            return np.zeros(model.vector_size)

        text_vector = np.mean(model[tokens], axis=0)
        return np.array(text_vector)


    # ## Pre-trained Word2Vec and ML algorithms


    import xgboost

    def return_score_xgb(sample_size, num_classes):

        train, test = gen_sample(sample_size, num_classes)

        X_train = train['Text'].values
        y_train = train['Class'].values
        X_test = test['Text'].values
        y_test = test['Class'].values

        X_train_mean = np.array([transform_sentence(x, model) for x in X_train])
        X_test_mean = np.array([transform_sentence(x, model) for x in X_test])

        #     XG Boost
        clf = xgboost.XGBClassifier()
        clf.fit(X_train_mean, y_train)

        if not os.path.exists('./pkl_objects'):
            os.mkdir('./pkl_objects')

        joblib.dump(le, './pkl_objects/labelencoder.pkl')
        joblib.dump(clf, './pkl_objects/clf.pkl')

        y_pred = clf.predict(X_test_mean)

        return accuracy_score(y_pred, y_test)


    all_accuracy_xgb = {2:[],3:[],4:[],5:[],6:[],7:[]}

    for num_samples in range(1, 40):

        for num_cl in range(2, 7):

            all_accuracy_xgb[num_cl].append(return_score_xgb(num_samples,num_cl))
