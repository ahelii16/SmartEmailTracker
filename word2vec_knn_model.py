import pandas as pd
import numpy as np
from random import seed
from random import sample

seed(42)
np.random.seed(42)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import gensim.downloader as api
from gensim.models.keyedvectors import Word2VecKeyedVectors

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy import spatial

import re
import spacy
import en_core_web_sm

model = api.load('word2vec-google-news-300')

def get_only_chars(line):    
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]


    nlp_ = en_core_web_sm.load()
    nlp = spacy.load('en')
    my_stop = ["'d", "'ll", "'m", "'re", "'s", "'ve",'a','cc','subject','http', 'gbp', 'usd', 'eur', 'inr', 'cad', 'thanks', "acc", "id", 'account', 'regards', 'hi', 'hello', 'thank you', 'greetings', 'about','above', 'across','after','afterwards','against','alone','along','already','also','although','am','among', 'amongst','amount','an','and','another','any','anyhow','anyone','anything','anyway','anywhere','are','around','as', 'at','be','became','because','become','becomes','becoming','been','before','beforehand','behind','being','below', 'beside','besides','between','both','bottom','but','by','ca','call','can','could','did', 'do', 'does', 'doing', 'down', 'due', 'during', 'each', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'every', 'everyone', 'everything', 'everywhere', 'fifteen', 'fifty', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'four', 'from', 'front', 'further', 'get', 'give', 'go', 'had', 'has', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'made', 'make', 'many', 'may', 'me', 'meanwhile', 'might', 'mine', 'more', 'moreover', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'quite', 'rather', 're', 'really', 'regarding', 'same', 'say', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'under', 'unless', 'until', 'up', 'upon', 'us', 'used', 'using', 'various', 'very', 'via', 'was', 'we', 'well', 'were', 'whatever', 'whence', 'whenever', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'whoever', 'whole', 'whom', 'whose', 'will', 'with', 'within', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', '‘d', '‘ll', '‘m', '‘re', '‘s', '‘ve', '’d', '’ll', '’m', '’re', '’s', '’ve']

    line=line.rstrip()
    line = re.sub(r'[^a-zA-Z]', ' ', line)
    line = " ".join([i for i in line.split()])
#     print(line)
    
    doc = nlp(line)
    
    normalized = " ".join(token.lemma_ for token in doc if token not in my_stop)
    
    doc = " ".join(token.orth_ for token in normalized if not token.is_punct | token.is_space)
    return doc

# num_classes = 2 # the number of classes we consider (since the dataset has many classes)
# sample_size = 3 # the number of labeled sampled we’ll require from the user


# # Generate samples that contains K samples of each class

# def gen_sample(sample_size, num_classes):

#     df_1 = df[(df["Class"] < num_classes+1)].reset_index().drop(["index"], axis=1).reset_index().drop(["index"], axis=1)
#     train = df_1[df_1["Class"] == np.unique(df_1['Class'])[0]].sample(sample_size)

#     train_index = train.index.tolist()

#     for i in range(1,num_classes):
#         train_2 = df_1[df_1["Class"] == np.unique(df_1['Class'])[i]].sample(sample_size)
#         train = pd.concat([train, train_2], axis=0)
#         train_index.extend(train_2.index.tolist())

#     test = df_1[~df_1.index.isin(train_index)]
#     # return test
#     return train, test


# Text processing (split, find token id, get embedidng)

def transform_sentence(text, model):

    """
    Mean embedding vector
    """

    def preprocess_text(raw_text, model=model):

        """ 
        Excluding unknown words and get corresponding token
        """

        raw_text = raw_text.split()

        return list(filter(lambda x: x in model.vocab, raw_text))

    tokens = preprocess_text(text)

    if not tokens:
        return np.zeros(model.vector_size)

    text_vector = np.mean(model[tokens], axis=0)

    return np.array(text_vector)


"""## Pre-trained Word2Vec and ML algorithms"""

import xgboost
import joblib
import os

def return_res_xgb(sample_size, num_classes=6):

    # train, test = gen_sample(sample_size, num_classes)

    # X_train = train['Text']
    # y_train = train['Class'].values
    # X_test = test['Text']
    # y_test = test['Class'].values

    X_train_mean = X_train.apply(lambda x : transform_sentence(x, model))
    X_test_mean = X_test.apply(lambda x : transform_sentence(x, model))

    X_train_mean = pd.DataFrame(X_train_mean)['Text'].apply(pd.Series)
    X_test_mean = pd.DataFrame(X_test_mean)['Text'].apply(pd.Series)

    # XG Boost
    clf = xgboost.XGBClassifier()
    clf.fit(X_train_mean, y_train)

    y_pred = clf.predict(X_test_mean)

    if not os.path.exists('./pkl_objects'):
        os.mkdir('./pkl_objects')
    
    joblib.dump(le, './pkl_objects/labelencoder.pkl')
    joblib.dump(clf, './pkl_objects/clf.pkl')

    return le.inverse_transform(y_pred - 1)

def inp(emailto, emailfrom, subj, bod):
    text = subj + " " + bod
    text = get_only_chars(text)

    text_mean = 