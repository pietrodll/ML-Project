# -*- coding: utf-8 -*-

from sklearn import cross_validation
from sklearn.metrics import r2_score, mean_squared_log_error, mean_absolute_error, mean_squared_error
import numpy as np

def crossValidation(X, y, classfunction, errorFunction=mean_absolute_error):
    kf = cross_validation.KFold(X.shape[0], n_folds=8)
    
    errors = [] # Variable containing errors for each fold
    scores = [] # Variable containing the R^2 scores for each fold
    totalInstances = 0
    totalError = 0
    
    for trainIndex, testIndex in kf:
        trainSet = X[trainIndex]
        testSet = X[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
            	
        predictedLabels = classfunction(trainSet, trainLabels, testSet)
        
        error = errorFunction(testLabels, predictedLabels)
        totalError += error*testIndex.shape[0]
        totalInstances += testIndex.shape[0]
        
        errors.append(error)
        scores.append(r2_score(testLabels, predictedLabels))
            
    print('Error : ', np.around(totalError/totalInstances, 5))
    print('Errors : ', np.around(errors, 2))
    print('Scores : ', np.around(scores, 2))
    
    return totalError/totalInstances