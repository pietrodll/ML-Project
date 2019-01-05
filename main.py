# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model, neural_network
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

def plotFeatureSelection(X, y):
    error = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        # Xnew = pre.PCA(X, k)
        error[k-1] = crossValidation(Xnew, y, KNNregression)
    error[-1] = crossValidation(X, y, KNNregression)
    plt.plot(list(range(1, X.shape[1]+1)), error,color='blue')
    
    error2 = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        # Xnew = pre.PCA(X, k)
        error2[k-1] = crossValidation(Xnew, y, linearRegression)
    error2[-1] = crossValidation(X, y, linearRegression)
    plt.plot(list(range(1, X.shape[1]+1)), error2, color='green')
    
    error3 = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        #Xnew = pre.PCA(X, k)
        error3[k-1] = crossValidation(Xnew, y, stochGrad)
    error3[-1] = crossValidation(X, y, stochGrad)
    plt.plot(list(range(1, X.shape[1]+1)), error3, color='red')
    
    error4 = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        #Xnew = pre.PCA(X, k)
        error4[k-1] = crossValidation(Xnew, y, neuralNetwork)
    error4[-1] = crossValidation(X, y, neuralNetwork)
    plt.plot(list(range(1, X.shape[1]+1)), error4, color='orange')
    
def displayResult(X,y,k):
    Xnew = pre.featureSelect(X,y,k)
    print("Neighbors : ")
    crossValidation(Xnew,y,KNNregression,display=True)
    print("Linear Regression : ")
    crossValidation(Xnew,y,linearRegression,display=True)
    print("Stochastic : ")
    crossValidation(Xnew,y,stochGrad,display=True)
    print("Neural network : ")
    crossValidation(Xnew,y,neuralNetwork,display=True)