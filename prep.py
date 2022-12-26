import numpy as np
import pandas as pd
from sklearn import preprocessing
import random

def prep(df, 
         axis: str,
         perc: int  = 100,
         fill_method: str = 'mean',
         scale: bool = True,
         scaler: str = preprocessing.MinMaxScaler()):
    
    if axis not in ['col', 'obs']:
        print('choose the axis: "col" or "obs"')
        return
        
    elif axis=='col':
        new_df = df.dropna(axis=1, how='any')
        return new_df
        
    assert fill_method in ['mean', 'median']
    assert perc in range(101)
    
    row, _ = df.shape
    null_index = []
    means = df.mean()
    
    for i in range(row):
        obs = df.loc[i]
        if obs.isnull().sum() != 0:
            null_index.append(i)
    
    tot_null = len(null_index)
    torm = int(perc * tot_null / 100)
    tofill = tot_null - torm
    num = random.choices(null_index, k=tofill)
    
    if fill_method=='mean':
        for i in num:
            df.loc[i] = df.loc[i].fillna(means)
    else:
        for i in num:
            df.loc[i] = df.loc[i].fillna(means)
    
    clean_df = df.dropna(axis=0)
    
    if scale == True:
        scaled_df = scaler.fit_transform(clean_df)
        return scaled_df
    else:
        return clean_df