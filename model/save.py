# -*- coding: utf-8 -*-
"""
@Time : 2022/1/6 21:32 
@Author : Chao Zhu
"""
import numpy as np
import pandas as pd


y_true = pd.Series(np.load('../result/y_true.npy'), name='y_true')
y_pred = pd.Series(np.load('../result/y_pred.npy'), name='y_pred')

y = pd.concat([y_true, y_pred], axis=1)
y.to_csv('../result/y_true_pred.csv')
print(y)
