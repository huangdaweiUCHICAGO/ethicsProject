import random
from scipy import spatial
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics

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

#replaces a random datapoint in column df[col] bounded by column min and column max
def randomReplace(df,col):
    minimum = min(list(df[col]))
    maximum = max(list(df[col]))
    randNum = random.uniform(minimum,maximum)
    randInd = random.randrange(0,len(list(df[col]))-1)
    df.loc[randInd,col] = randNum
    return df

#runs through iteration for a single column for i rounds
def listGen(col,i,xlist,ylist,func):
    accuracyList = []
    func.fit(xlist, ylist)
    predInit = func.predict(xlist)
    for x in range(0,i):
        func.fit(xlist, ylist)
        pred = func.predict(xlist)
        accuracyList.append(1 - spatial.distance.cosine(pred, predInit))
        xlist = randomReplace(xlist,col) 
    return accuracyList

#corrects for some of the noise and smooths out the data slightly
def rCorrect(l):
    if(l[1] < l[-1]):
        med = statistics.mean(l)
        for i in range(0,len(l)):
            diff = abs(l[i] - med)
            if(l[i] >= med):
                l[i] -= 2*diff
            else:
                l[i] += 2*diff
    return l

#smooths out the list by averaging closest neighbors in array.
def smoothList(l):
    if len(l) -1 <= 1:
        return l
    newList = []
    newList.append(l[0])
    for i in range(1, len(l)-1):
        meanR = (l[i-1]+l[i]+l[i+1])/3
        newList.append(meanR)
    newList.append(l[-1])
    return newList
   
#creates graph 
def createGraph(X,y,col,i,func,smooth):
    fig = plt.figure(figsize=(8,5))
    df = pd.DataFrame( columns=['variable','slope'])
    for x in col:
        xlist = X.copy()
        ylist = y.copy()
        retArr = rCorrect(listGen(x,i,xlist,ylist,func))
        if smooth == True:
            retArr = smoothList(retArr)
        slope = (retArr[-1] - retArr[0])/i
        if slope == 0:
            slope = 9999999999
        tempRow = {'variable':x, 'slope':slope}
        #append row to the dataframe
        df = df.append(tempRow, ignore_index=True)
        ax1 = fig.add_subplot(121)
        ax1.plot(retArr,label=x)
        ax1.legend()
    df = df.sort_values(by=['slope'], ascending=True)
    ax1.table(cellText=df.values,colWidths = [1]*len(df.columns),
          rowLabels=df.index,
          colLabels=df.columns,
          cellLoc = 'center', rowLoc = 'center',
          loc='right')

def locateTop(l,ideal):
    score = 0
    if len(l) < len(ideal):
        return score
    else:
        for x in l[:len(ideal)]:
            if x in ideal:
                score += 1
    return score/len(ideal)

#scores the accuracy
def accuracyScore(X,y,col,i,func,ideal,iteration):
    df = pd.DataFrame( columns=['variable','slope'])
    accScore = 0
    for ix in range(iteration):
        for x in col:
            xlist = X.copy()
            ylist = y.copy()
            retArr = rCorrect(listGen(x,i,xlist,ylist,func))
            slope = (retArr[-1] - retArr[0])/i
            if slope == 0:
                slope = 9999999999
            tempRow = {'variable':x, 'slope':slope}
            #append row to the dataframe
            df = df.append(tempRow, ignore_index=True)
        df = df.sort_values(by=['slope'], ascending=True)
        accScore += locateTop(df['variable'],ideal)
    accScore /= iteration
    return accScore