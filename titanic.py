# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:17:05 2019

@author: Greg13
"""
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
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
compute_score(lr, X, y) # 0.674548857768335

# Etude des variables
survived = train[train.Survived == 1]
dead = train[train.Survived == 0]


def plot_hist(feature, bins = 20):
    x1 = dead[feature].dropna()
    x2 = survived[feature].dropna()
    plt.hist([x1,x2], label=['Victime', 'Survivant'], bins = bins, color = ['pink', 'skyblue'])
    plt.legend(loc = 'upper left')
    plt.title('distribution relative de %s' %feature)
    plt.show()
    
plot_hist('Pclass')

# split des class en 3 catégorie
def parse_model_1(X):
    target = X.Survived
    class_dummies = pd.get_dummies(X['Pclass'], prefix='split_Pclass')
    X = X[['Fare', 'SibSp', 'Parch']]
    X = X.join(class_dummies)    
    return  X, target

X, y = parse_model_1(train.copy())
lr = LogisticRegression()
compute_score(lr, X, y) #0.6926591973081655


# Poids pour chaque variable de la régression logistique (poids + augmente proba et - diminue proba)
lr = LogisticRegression()
lr.fit(X,y)
print(lr.coef_)  
#     Fare        SibSp        Parch      Pclass1     Pclass2     Pclass3
# [[ 0.00669907 -0.150896    0.23357229  0.3730938   0.100852   -0.85258357]]


# Ajout sexe et age avec remplissage des variables manquante pour l'age avec la médiane
def parse_model_2(X):
    target = X.Survived
    to_dummy = ['Pclass', 'Sex']
    for dum in to_dummy :
        split_temp = pd.get_dummies(X[dum], prefix='split_'+dum)
        for col in split_temp :
            X[col] = split_temp[col]
        del X[dum]
    X['Age'] = X.Age.fillna(X.Age.median())
    to_del = ['Name', 'Cabin', 'Embarked', 'Survived', 'Ticket']
    for col in to_del : del X[col]
    return  X, target

X, y = parse_model_2(train.copy())
lr = LogisticRegression()
compute_score(lr, X, y) #0.7868160254657532

plot_hist('Age')

# Ajout info est-ce un enfant
def parse_model_2bis(X):
    target = X.Survived
    to_dummy = ['Pclass', 'Sex']
    for dum in to_dummy :
        split_temp = pd.get_dummies(X[dum], prefix='split_'+dum)
        for col in split_temp :
            X[col] = split_temp[col]
        del X[dum]
    X['Age'] = X.Age.fillna(X.Age.median())
    X['is_child'] = X.Age < 8
    to_del = ['Name', 'Cabin', 'Embarked', 'Survived', 'Ticket']
    for col in to_del : del X[col]
    return  X, target

X, y = parse_model_2bis(train.copy())
lr = LogisticRegression()
compute_score(lr, X, y) #0.800274134181057



