# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt


def setMonths2(data):
    """
    data: pandas DataFrame
    Replaces the month feature by 12 features, one for each month, which can only be 1 or 0.
    """
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for month in months:
        new_column = pd.Series(0, index=data.index)
        new_column[data.month == month] = 1
        data[month] = new_column
    data.drop('month', axis=1, inplace=True)


def setDays2(data):
    """
    data: pandas DataFrame
    Replaces the month feature by 12 features, one for each month, which can only be 1 or 0.
    """
    days = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
    for day in days:
        new_column = pd.Series(0, index=data.index)
        new_column[data.day == day] = 1
        data[day] = new_column
    data.drop('day', axis=1, inplace=True)

def setWeekend(data):
    """
    data: pandas DataFrame
    Replaces the day feature by a weekend feature, whose value is 1 if the day is Saturday or Sunday, 0 otherwise.
    """
    new_col = pd.Series(0, index=data.index)
    new_col[data.day == 'sat'] = 1
    new_col[data.day == 'sun'] = 1
    data['weekend'] = new_col
    # data.drop('day', axis=1, inplace=True)

"""
def loadData(path='data/forestfires.csv', process='none', shuffle=True):
    data = pd.read_csv(path)
    if shuffle:
        data = data.reindex(np.random.permutation(data.index))
    data = data[data.area > 0] #We only take data where the fire has happened
    X, y = data.iloc[:, :-1], data.iloc[:, -1] # Split attributes and labels
    setDays(X)
    setMonths(X)
    if process == 'scale':
        scaleData(X)
    elif process == 'normalize':
        normalizeData(X)
    X = np.array(X) # Turn the DataFrames into Numpy Arrays
    y = np.array(y)
    return X, y
"""

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
        if M[col] != m[col]:
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


def PCA(A, y, k):
    """
    A : Numpy Array, k : integer
    Performs PCA on A
    """
    M = np.tile(np.average(A, axis=0), (A.shape[0], 1)) # Mean of the columns
    C = A - M
    W = np.dot(np.transpose(C), C)
    _, eigvec = np.linalg.eigh(W)
    eigvec = eigvec[:,::-1] # eigenvalues in ascending order : colums of U must be reversed
    Uk = eigvec[:,:k]
    return np.dot(A, Uk)


def featureSelect(X, y, j):
    Xnew = SelectKBest(f_regression, k=j).fit_transform(X, y)
    return Xnew


def plotData(X, y, features, sepFig=False):
    a = np.array(y)
    if features == None:
        colList = X.columns
    else:
        colList = features
    for col in colList:
        if sepFig:
            plt.figure()
            plt.title(col)
        A = np.array(X[col])
        plt.plot(A, np.log(a), '+', label=col)
        if len(colList) == 1:
            plt.title(col)
        


X = pd.read_csv('data/forestfires.csv') # Load the dataset
X = X[X.area > 0] # We only take data where the fire has happened
y = X['area'] # Extract labels from the dataset
X.drop('area', axis=1, inplace=True) # Removing the label column from X
# setMonths2(X)
setWeekend(X)

# scaleData(X)
# plotData(X, y)