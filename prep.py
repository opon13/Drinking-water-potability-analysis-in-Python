import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random



def prep(data, 
         target: str = None,
         axis: str = 'obs',
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
    
    df = data.copy()
    row, _ = df.shape
    null_index = []
    
    if target != None:
        assert target in list(df.columns)
        variables = list(df.columns)
        variables.remove(target)
        means = df.groupby(target)[variables].mean()
        medians = df.groupby(target)[variables].median()
    else:
        means = df.mean()
        medians = df.median()
    
    for i in range(row):
        obs = df.loc[i]
        if obs.isnull().sum() > 0:
            null_index.append(i)
    
    tot_null = len(null_index)
    to_rm = int(perc * tot_null / 100)
    to_fill = tot_null - to_rm
    fill = []
    
    for i in range(to_fill):
        x = random.choice(null_index)
        fill.append(x)
        null_index.remove(x)
    
    if fill_method=='mean':
        if target != None:
            for i in fill:
                for j in means.index:
                    if df.loc[i][target]==j:
                        df.loc[i] = df.loc[i].fillna(means.loc[j])
        else:
            for i in fill:
                df.loc[i] = df.loc[i].fillna(means)
    else:
        if target != None:
            for i in fill:
                for j in medians.index:
                    if df.loc[i][target]==j:
                        df.loc[i] = df.loc[i].fillna(medians.loc[j])
        else:
            for i in fill:
                df.loc[i] = df.loc[i].fillna(medians)
    
    clean_df = df.dropna(axis=0, how='any').reset_index(drop=True)
    
    if scale == True:
        scaled_df = scaler.fit_transform(clean_df)
        return scaled_df
    else:
        return clean_df
    


def split(df,
          target_index: int,
          validation: bool = True,
          perc_train: float or int = 0.6, 
          random_seed: int = None):
    
    assert target_index in range(df.shape[1])
    
    variables_index = [x for x in range(df.shape[1])]
    variables_index.remove(target_index)
    X = df.iloc[:, variables_index]
    y = df.iloc[:, target_index]

    print('BEFORE SPLITTING: \n')
    print('X shape: ', np.shape(X))
    print('y shape: ', np.shape(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (1-perc_train), random_state = random_seed)

    print('\nAFTER SPLITTING: ')
    print('X_train shape: ', np.shape(X_train))
    print('y_train shape: ', np.shape(y_train))
    print('X_test shape: ', np.shape(X_test))
    print('y_test shape: ', np.shape(y_test))
    
    if validation == True:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state = random_seed)
        print('X_val shape: ', np.shape(X_val))
        print('y_val shape: ', np.shape(y_val))
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test