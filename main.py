# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
from validation import crossValidation
import preprocessing as pre

X, y = pre.loadData(process='normalize', logLabels=False)
# X = X[2:]


def KNNregression(trainSet, trainLabels, testSet):
    N = len(trainSet)
    neigh = KNeighborsRegressor(n_neighbors=int(sqrt(N)), weights='distance')
    neigh.fit(trainSet, trainLabels)
    y = neigh.predict(testSet)
    return y

# def linearRegression(trainSet, trainLabels, testSet):


def plotFeatureSelection(X, y):
    error = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        # Xnew = pre.PCA(X, k)
        error[k-1] = crossValidation(Xnew, y, KNNregression)
    error[-1] = crossValidation(X, y, KNNregression)
    plt.plot(list(range(1, X.shape[1]+1)), error)