# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:17:05 2019

@author: Greg13
"""
import pandas as pd
train = pd.read_csv('data/train.csv', sep = ',')
train.head(10)
train.set_index('PassengerId', inplace=True, drop=True)

train.columns





