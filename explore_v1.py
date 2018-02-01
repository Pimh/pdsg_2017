# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:11:28 2017

@author: PimH
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

def main():
    ''' Load data using pandas '''
    data = pd.read_json('train.json', orient='columns')
    #print (data.head(5))
    #print(data.describe())
    
    
    ''' Clean up data '''
    # print columns with null values
    for column in data.columns:
        if np.any(pd.isnull(data[column])):
            print(column)
    
    # display data type of each column
    for column in data.columns:
        print column, data[column].dtype
        
    # convert interest levels to categorical values
    data['interest_level'] = data['interest_level'].astype('category')
    
    # convert date created into numerical values
    dtime_ref = datetime.datetime.strptime('2017-03-06 14:40:55', \
    '%Y-%m-%d %H:%M:%S')
    for idx in data.index:
        dtime = datetime.datetime.strptime(data.loc[idx,'created'], \
        '%Y-%m-%d %H:%M:%S')
        dtime_delta = dtime_ref - dtime
        data.loc[idx, 'delta_dtime_created'] = dtime_delta.days
        
        # extract day of the week each listing was created    
        data.loc[idx, 'day_of_week'] = dtime.weekday()
        
        # simple processing: photos > # of photos
        data.loc[idx, 'nPhoto'] = len(data.loc[idx, 'photos'])
    
    # convert to categorical values: building_id, manager_id
    data['building_id'] = data['building_id'].astype('category')
    data['manager_id'] = data['manager_id'].astype('category')
    data['day_of_week'] = data['day_of_week'].astype('category')
    
    # some text processing: features, description, address
    
    
    ''' Copy the dataframe and only include the simple features
        bathrooms, bedrooms, latitude, longitude, price, delta_dtime_created, 
        day of the week, number of photos '''
    X = data[['bathrooms', 'bedrooms', 'latitude', 'longitude', \
        'price', 'delta_dtime_created', 'day_of_week', 'nPhoto']].copy()
    y = np.asarray(data.loc[:,'interest_level'])
    
    
    ''' Split the dataset into test and train sets, and train a model '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, \
        random_state = 0)
        
    # First try a random forest model
    RF_clf = RandomForestClassifier().fit(X_train, y_train)
    y_pred_prob = RF_clf.predict_proba(X_test)
    logLoss = log_loss(y_test, y_pred_prob)
    print 'log loss = ', logLoss
    
    y_pred = RF_clf.predict(X_test)
    cnf = confusion_matrix(y_test, y_pred)
    class_names = ['Low', 'Medium', 'High']
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()

# Function for plotting confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


main()