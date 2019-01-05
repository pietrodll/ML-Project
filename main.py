# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model, neural_network
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
    plt.plot(np.arange(1, X.shape[1]+1), error,color='blue')
    
    error2 = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        # Xnew = pre.PCA(X, k)
        error2[k-1] = crossValidation(Xnew, y, linearRegression)
    error2[-1] = crossValidation(X, y, linearRegression)
    plt.plot(np.arange(1, X.shape[1]+1), error2, color='green')
    
    error3 = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        #Xnew = pre.PCA(X, k)
        error3[k-1] = crossValidation(Xnew, y, stochGrad)
    error3[-1] = crossValidation(X, y, stochGrad)
    plt.plot(np.arange(1, X.shape[1]+1), error3, color='red')
    
    error4 = [0]*X.shape[1]
    for k in range(1, X.shape[1]):
        Xnew = pre.featureSelect(X, y, k)
        #Xnew = pre.PCA(X, k)
        error4[k-1] = crossValidation(Xnew, y, neuralNetwork)
    error4[-1] = crossValidation(X, y, neuralNetwork)
    plt.plot(np.arange(1, X.shape[1]+1), error4, color='orange')
    
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
