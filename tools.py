import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from  sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
import random



def prep(data, 
         target: str = None,
         axis: str = 'obs',
         perc: int  = 100,
         fill_method: str = 'mean',
         scale: bool = True,
         scaler = preprocessing.MinMaxScaler(),
         random_state: int = None):
    
    if axis not in ['col', 'obs']:
        print('choose the axis: "col" or "obs"')
        return
        
    elif axis=='col':
        new_df = df.dropna(axis=1, how='any')
        return new_df

    if random_state != None:
        random.seed(random_state)

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
        target_df = clean_df[target]
        features = clean_df.drop(target, axis=1)
        features_name = features.columns
        scaled_df = scaler.fit_transform(features)
        scaled_df = pd.DataFrame(scaled_df, columns = features_name)
        scaled_df[target] = target_df
        scaled_df = np.array(scaled_df)
        return scaled_df
    else:
        return clean_df
    


def split(df,
          target_index: int,
          validation: bool = True,
          perc_train: float or int = 0.6, 
          random_seed: int = None,
          verbose=True):
    
    assert target_index in range(df.shape[1])
    
    variables_index = [x for x in range(df.shape[1])]
    variables_index.remove(target_index)
    X = df[:, variables_index]
    y = df[:, target_index]
    if(verbose==True):
        print('BEFORE SPLITTING: \n')
        print('X shape: ', np.shape(X))
        print('y shape: ', np.shape(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (1-perc_train), random_state = random_seed)
    if(verbose==True):
        print('\nAFTER SPLITTING: ')
        print('X_train shape: ', np.shape(X_train))
        print('y_train shape: ', np.shape(y_train))
        print('X_test shape: ', np.shape(X_test))
        print('y_test shape: ', np.shape(y_test))
    
    if validation == True:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state = random_seed)
        if(verbose==True):
            print('X_val shape: ', np.shape(X_val))
            print('y_val shape: ', np.shape(y_val))
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test


def evaluate(model,
    features, 
    labels,
    cv: bool = False,
    k_fold: int = 5, 
    conf_matrix: bool = False):

    if cv==True:
        metrics = cross_validate(model, features, labels, cv = k_fold, scoring=['accuracy', 'recall', 'precision', 'f1'])
        accuracy = np.mean(metrics['test_accuracy'])
        recall = np.mean(metrics['test_recall'])
        precision = np.mean(metrics['test_precision'])
        f1 = np.mean(metrics['test_f1'])
    else:
        predictions = model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, zero_division=0)
        precision = precision_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        if conf_matrix==True:
            print('Confusion matrix: ')
            cm = confusion_matrix(labels, predictions)
            cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
            cm_display.plot()
            plt.show()
    
    print('Model Performance: \n')
    print('accuracy = {:0.2f}%.'.format(accuracy*100))
    print('recall = {:0.2f}%.'.format(recall*100))
    print('precision = {:0.2f}%.'.format(precision*100))
    print('f1_score = {:0.2f}%.'.format(f1*100))

    return accuracy, recall, precision, f1

def improvements(previous_metrics, new_metrics, previous_model, new_model):
    assert type(previous_model)==str and type(new_model)==str
    print('\nImprovements of the '+ new_model +' model over the '+ previous_model +' model:\n')
    metrics = []
    for i in range(4):
        j = 100 * (new_metrics[i] - previous_metrics[i])
        metrics.append(j)
    print('improvement in accuracy: {:0.2f} %'.format(metrics[0]))
    print('improvement in recall: {:0.2f} %'.format(metrics[1]))
    print('improvement in precision: {:0.2f} %'.format(metrics[2]))
    print('improvement in  f1-score: {:0.2f} %'.format(metrics[3]))
    