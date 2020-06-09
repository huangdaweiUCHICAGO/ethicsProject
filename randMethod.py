import random
from scipy import spatial
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from scipy import stats

def randomReplace(df,col):
    minimum = min(list(df[col]))
    maximum = max(list(df[col]))
    randNum = random.uniform(minimum,maximum)
    randInd = random.randrange(0,len(list(df[col]))-1)
    df.loc[randInd,col] = randNum
    return df

def listGen(col,i,xlist,ylist,func):
    accuracyList = []
    for x in range(0,i):
        func.fit(xlist, ylist)
        pred = func.predict(xlist)
        accuracyList.append(1 - spatial.distance.cosine(pred, list(ylist)))
        xlist = randomReplace(xlist,col) 
    return accuracyList

def createGraph(X,y,col,i,func):
    fig = plt.figure(figsize=(8,5))
    df = pd.DataFrame( columns=['variable','slope'])
    for x in col:
        xlist = X.copy()
        ylist = y.copy()
        retArr = listGen(x,i,xlist,ylist,func)
        slope, intercept, r_value, p_value, std_err = stats.linregress(retArr, range(0,i))
        if slope == 0:
            slope = -9999999999999
        tempRow = {'variable':x, 'slope':-1 * abs(slope)}
        #append row to the dataframe
        df = df.append(tempRow, ignore_index=True)
        ax1 = fig.add_subplot(121)
        ax1.plot(retArr,label=x)
        ax1.legend()
    df = df.sort_values(by=['slope'], ascending=False)
    ax1.table(cellText=df.values,colWidths = [1]*len(df.columns),
          rowLabels=df.index,
          colLabels=df.columns,
          cellLoc = 'center', rowLoc = 'center',
          loc='right')