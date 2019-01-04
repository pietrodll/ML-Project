# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_csv('data/forestfires.csv')

def setDays(data):
    days = {'sun':7, 'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6}
    a = pd.Series(index=data.index)
    for d in days.keys():
        a[data.day == d] = days[d]
    data.day = a

def setMonths(data):
    months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    a = pd.Series(index=data.index)
    for m in months.keys():
        a[data.month == m] = months[m]
    data.month = a

setMonths(data)
setDays(data)