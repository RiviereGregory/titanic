# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:17:05 2019

@author: Greg13
"""
import pandas as pd
import statistics as stat
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('data/train.csv', sep = ',')
train.head(10)
train.set_index('PassengerId', inplace=True, drop=True)

train.columns


# Premier modèle
def parse_model_0(X):
    target = X.Survived
    X = X[['Fare', 'SibSp', 'Parch']]
    return  X, target

X, y = parse_model_0(train.copy())

# Moyenne de plusieurs validations croisées
def compute_score(clf, X, y):
    xval = cross_val_score(clf,X, y, cv = 5)
    return stat.mean(xval)

lr = LogisticRegression()
compute_score(lr, X, y)



