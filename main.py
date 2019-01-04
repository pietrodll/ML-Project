# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
from validation import crossValidation
import preprocessing as pre

X, y = pre.loadData(process='normalize')

def regression(trainSet, trainLabels, testSet):
    N = len(trainSet)
    neigh = KNeighborsRegressor(n_neighbors=int(sqrt(N)), weights='distance')
    neigh.fit(trainSet, trainLabels)
    y = neigh.predict(testSet)
    return y

crossValidation(X, y, regression)