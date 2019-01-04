# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def loadData(path='data/forestfires.csv', process='none'):
    data = pd.read_csv(path)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    setDays(X)
    setMonths(X)
    if process == 'scale':
        scaleData(X)
    elif process == 'normalize':
        normalizeData(X)
    return np.array(X), np.array(y)

def setDays(data):
    """
    data : pandas DataFrame
    Replaces the name of the day of the week by a number
    """
    days = {'sun':7, 'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6}
    a = pd.Series(index=data.index)
    for d in days.keys():
        a[data.day == d] = days[d]
    data.day = a

def setMonths(data):
    """
    data : pandas DataFrame
    Replaces the name of the month by a number
    """
    months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    a = pd.Series(index=data.index)
    for m in months.keys():
        a[data.month == m] = months[m]
    data.month = a

def scaleData(X):
    """
    X : pandas DataFrame with only numerical values
    This function linearly scales the features of X, to make each value lie between 0 and 1
    """
    M, m = X.max(), X.min()
    for col in X.columns:
        X[col] = (X[col] - m[col])/(M[col] - m[col])

def normalizeData(X):
    """
    X : pandas DataFrame with only numerical values
    This function linearly scales the features of X, to make it centered with a unitary standard deviation
    """
    M, S = X.mean(), X.std()
    for col in X.columns:
        if S[col] == 0:
            X[col] = 0
        else:
            X[col] = (X[col] - M[col])/S[col]