# -*- coding: utf-8 -*-

from sklearn import cross_validation
from sklearn.metrics import r2_score, mean_squared_log_error, mean_absolute_error, mean_squared_error

def crossValidation(X, y, classfunction, errorFunction=mean_absolute_error):
    kf = cross_validation.KFold(X.shape[0], n_folds=10)
    
    errors = [] # Variable containing errors for each fold
    scores = [] # Variable containing the R^2 scores for each fold
    
    for trainIndex, testIndex in kf:
        trainSet = X[trainIndex]
        testSet = X[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
            	
        predictedLabels = classfunction(trainSet, trainLabels, testSet)
        
        errors.append(errorFunction(testLabels, predictedLabels))
        scores.append(r2_score(testLabels, predictedLabels))
            
    print('Errors : ', errors)
    print('Scores : ', scores)