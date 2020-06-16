import numpy as np
import pandas as pd

data=[[11, 7, 8, 80000], [0, 8, 9, 50000], [0, 8, 6, 45000], [5, 6, 7, 60000], [2, 10, 10, 65000], [7, 9, 6, 70000], [3, 7, 10, 62000], [10, 7.85714, 7, 72000]]

df = pd.DataFrame(data, columns = ['Experience','Test_score','interview_score','Salary',])

X=df.iloc[:,:3]
Y=df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X,Y)

