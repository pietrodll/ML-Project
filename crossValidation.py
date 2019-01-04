# -*- coding: utf-8 -*-

from sklearn import cross_validation

def crossValidation(X, y, classfunction):
    kf = cross_validation.KFold(X.shape[0], n_folds=10)
    
    totalInstances = 0 # Variable that will store the total intances that will be tested  
    totalCorrect = 0   # Variable that will store the correctly predicted intances  
    
    for trainIndex, testIndex in kf:
        trainSet = X[trainIndex]
        testSet = X[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
            	
        predictedLabels = classfunction(trainSet, trainLabels, testSet)
    
        correct = 0	
        for i in range(testSet.shape[0]):
            if predictedLabels[i] == testLabels[i]:
                correct += 1
            
        print('Accuracy : ' + str(float(correct)/(testLabels.size)))
        totalCorrect += correct
        totalInstances += testLabels.size
    
    print('Total Accuracy : ' + str(totalCorrect/float(totalInstances)))