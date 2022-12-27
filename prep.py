import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random



def prep(df, 
         axis: str,
         perc: int  = 100,
         fill_method: str = 'mean',
         scale: bool = True,
         scaler = preprocessing.MinMaxScaler()):
    
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
    
    clean_df = df.dropna(axis=0, how='any').reset_index(drop=True)
    
    if scale == True:
        scaled_df = scaler.fit_transform(clean_df)
        return scaled_df
    else:
        return clean_df


def splitting_func(df,perctrain=0.60,perctest=0.50):
    X_water = df[:,0:8]
    y_water = df[:,9]

    print('BEFORE SPLITTING: \n')
    print('X_water0 shape: ', np.shape(X_water))
    print('y_water0 shape: ', np.shape(y_water))

    X_train, X_test, y_train, y_test = train_test_split(X_water, y_water, test_size=1-perctrain, random_state=13) # 60% of total values for train
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=13) # 20% of total values for both validation and test

    print('\nAFTER SPLITTING: ')
    print('X_train0 shape: ', np.shape(X_train))
    print('X_val0 shape: ', np.shape(X_val))
    print('X_test0 shape: ', np.shape(X_test))
    print('y_train0 shape: ', np.shape(y_train))
    print('y_val0 shape: ', np.shape(y_val))
    print('y_test0 shape: ', np.shape(y_test))
    
    return(X_train,X_val, X_test,y_train, y_val, y_test)