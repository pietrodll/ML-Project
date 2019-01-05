# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model, neural_network, svm
from validation import crossValidation, plotFeatureSelection
import preprocessing as pre

X, y = pre.loadData(process='scale', shuffle=True)
# X = pre.featureSelect(X, y, 9)

def KNNregression(trainSet, trainLabels, testSet):
    N = len(trainSet)
    trainLabels = np.log(trainLabels) # logarithmic function
    neigh = KNeighborsRegressor(n_neighbors=int(np.sqrt(N)), weights='distance')
    neigh.fit(trainSet, trainLabels)
    y = neigh.predict(testSet)
    return np.exp(y)

def SVMregression(X, y, Xtest):
    clf = svm.SVR(kernel='rbf')
    clf.fit(X, y)
    ytest = clf.predict(Xtest)
    return ytest
    
def linearRegression(trainSet, trainLabels, testSet):
    regr = linear_model.LinearRegression()
    regr.fit(trainSet,trainLabels)
    return regr.predict(testSet)

def stochGrad(trainSet, trainLabels, testSet):
    regr = linear_model.SGDRegressor()
    regr.fit(trainSet,trainLabels)
    return regr.predict(testSet)

def neuralNetwork(trainSet, trainLabels, testSet):
    neu=neural_network.MLPRegressor()
    neu.fit(trainSet,trainLabels)
    return neu.predict(testSet)

regressors = [KNNregression, SVMregression, linearRegression, stochGrad]

def compareRegressors(X, y, regressors, featureFunction=pre.featureSelect):
    plt.figure()
    for func in regressors:
        plotFeatureSelection(X, y, func, featureFunction)
    plt.legend()
    plt.show()

def displayResult(X,y,k, regressors, featureFunction):
    Xnew = featureFunction(X, y, k)
    for func in regressors:
        print(func.__name__ + " : ")
        crossValidation(Xnew, y, func, display=True)
