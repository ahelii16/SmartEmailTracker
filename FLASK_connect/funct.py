import pickle
import numpy as np


def add(n,m):
    s=n+m
    return s


model=pickle.load(open('test_model.pkl','rb'))


def pred(a,b,c):
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    c = np.asarray(c, dtype='float64')
    p=model.predict([[a,b,c]])
    res=int(p)
    return res

k=pred('3','7','9')
print(k)