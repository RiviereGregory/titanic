# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:17:05 2019

@author: Greg13
"""
import pandas as pd
import numpy as np
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

lr.fit(X,y)
print(lr.coef_) 
#     Age         SibSp        Parch      Fare         Pclass1     Pclass2   Pclass3     Female      Male        enfant
# [[-0.02230836 -0.42181314 -0.19815951  0.004332    1.06824222  0.1841089 -0.8252507   1.59541268 -1.16831225  1.72150237]]
# is_child est descriminant car poid + important


# Random Forest
from sklearn.ensemble import RandomForestClassifier

X, y = parse_model_2(train.copy())
fr = RandomForestClassifier()
compute_score(fr, X, y) # 0.8070154235053925


def clf_importance(X, clf):
    import pylab as pl
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    pl.title('Feature importance')
    for tree in clf.estimators_:
        pl.plot(range(X.shape[1]), tree.feature_importances_[indices], 'r')
        pl.plot(range(X.shape[1]), importances[indices], 'b')
        pl.show()
    for f in range(X.shape[1]):
        print('%d. feature : %s (%f)' % (f + 1, X.columns[indices[f]] , importances[indices[f]]))

rf = RandomForestClassifier()
rf.fit(X, y)
clf_importance(X, rf)

#1. feature : Age (0.257123)
#2. feature : Fare (0.257019)
#3. feature : split_Sex_male (0.239873)
#4. feature : split_Sex_female (0.065598)
#5. feature : split_Pclass_3 (0.051683)
#6. feature : Parch (0.043072)
#7. feature : SibSp (0.042366)
#8. feature : split_Pclass_1 (0.024831)
#9. feature : split_Pclass_2 (0.018435)
        
# Utilisation des autres variable (Name)
X = train.copy()
X['title'] = X.Name.map(lambda x : x.split(',')[1].split('.')[0])
X['surname'] = X.Name.map(lambda x : '(' in x)

# Ajout name et cabin
def parse_model_4(X):
    target = X.Survived
    X['title'] = X.Name.map(lambda x : x.split(',')[1].split('.')[0])
    X['surname'] = X.Name.map(lambda x : '(' in x)
    X['Cabin'] = X.Cabin.map(lambda x : x[0] if not pd.isnull(x) else -1)
    to_dummy = ['Pclass', 'Sex', 'title', 'Embarked', 'Cabin']
    for dum in to_dummy :
        split_feature = pd.get_dummies(X[dum], prefix='split_'+dum)
        X = X.join(split_feature)
        del X[dum]
    X['Age'] = X.Age.fillna(X.Age.median())
    X['is_child'] = X.Age < 8
    to_del = ['Name', 'Survived', 'Ticket']
    for col in to_del : del X[col]
    return  X, target

X, y = parse_model_4(train.copy())
lr = LogisticRegression()
compute_score(lr, X, y) #0.8238448861562948

lr.fit(X,y)
print(lr.coef_) 

