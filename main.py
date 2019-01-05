# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
from validation import crossValidation
import preprocessing as pre
from sklearn import svm

X, y = pre.loadData(process='normalize', shuffle=True)
# X = pre.featureSelect(X, y, 9)

def plotData(X, y):
    for i in range(X.shape[1]):
        plt.figure()
        plt.plot(X[:, i], np.log(y), '+')

def KNNregression(trainSet, trainLabels, testSet):
    N = len(trainSet)
    trainLabels = np.log(trainLabels) # logarithmic function 
    neigh = KNeighborsRegressor(n_neighbors=int(sqrt(N)), weights='distance')
    neigh.fit(trainSet, trainLabels)
    y = neigh.predict(testSet)
    return np.exp(y)

def SVMregression(X, y, Xtest):
    clf = svm.SVR(kernel='poly')
    clf.fit(X, y)
    ytest = clf.predict(Xtest)
    return ytest

# def linearRegression(trainSet, trainLabels, testSet):
    

def plotFeatureSelection(X, y, regressionFunction):
    error = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        # Xnew = pre.PCA(X, k)
        error[k-1] = crossValidation(Xnew, y, regressionFunction)
    error[-1] = crossValidation(X, y, KNNregression)
    plt.plot(np.arange(1, X.shape[1]+1), error)