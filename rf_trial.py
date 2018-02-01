#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 23:27:58 2017

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
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict

def cal_multi_logloss(RF_clf, X_train, X_test, y_train, y_test):
    RF_clf.fit(X_train, y_train)
    y_pred_prob = RF_clf.predict_proba(X_test)
    logLoss = log_loss(y_test, y_pred_prob)
    print 'log loss = ', logLoss
    return logLoss

def cal_RF_confusion_matrix(RF_clf, X_train, X_test, y_train, y_test):
    # Calculate confusion matrix
    RF_clf.fit(X_train, y_train)
    y_pred_RF = RF_clf.predict(X_test)
    class_names = ['low', 'medium', 'high']
    cnf = confusion_matrix(y_test, y_pred_RF, labels = class_names)
    print 'RF confusion matrix'
    print cnf
    
    return cnf

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
        cm[i,j] = round(cm[i,j], 2)
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def cal_RF_cvscore(X_train, y_train):    
    # Calculate cross-validation score
    RF_clf = RandomForestClassifier(warm_start = True, 
                                     oob_score = True, 
                                     max_features = 'log2')
    RF_clf.set_params(n_estimators = 100)
    score = cross_val_score(RF_clf, X_train, y_train, cv = 5)
    print 'RF cross-validation score: ', score
    
    return score, RF_clf
    
def opt_RF_parameters(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, \
    random_state = 0)
    RANDOM_STATE = 123

    RF_clfs = [('Max features = sqrt', RandomForestClassifier(warm_start = True, 
                                     oob_score = True, 
                                     max_features = 'sqrt',
                                     random_state = RANDOM_STATE)), 
                ('Max features = log2', RandomForestClassifier(warm_start = True, 
                                     oob_score = True, 
                                     max_features = 'log2',
                                     random_state = RANDOM_STATE)),
                ('Max features = none', RandomForestClassifier(warm_start = True, 
                                     oob_score = True, 
                                     max_features = None,
                                     random_state = RANDOM_STATE))]
    
    n_estimator_min = 10
    n_estimator_max = 150
    error_rate = {RF_clfs[0][0]: [], RF_clfs[1][0]: [], RF_clfs[2][0]: [] }
    error_rate = OrderedDict(error_rate)
    
    for label, clf in RF_clfs:
        for n_est in range(n_estimator_min, n_estimator_max + 1, 10):
            # Set n_estimator
            clf.set_params(n_estimators = n_est)
            
            # Fit the classifiers to X,y
            clf.fit(X_train, y_train)
            
            # Calculate oob error
            oob_err = 1 - clf.oob_score_
            
            # Store oob error in the error dictionary
            error_rate[label].append(oob_err)
            
    # Plot "OOB error rate" vs. "n_estimators"
    n_estimator_range = range(n_estimator_min, n_estimator_max + 1, 10)
    for label, err in error_rate.items():
        plt.plot(n_estimator_range, error_rate[label], label = label)
        
    plt.xlim(n_estimator_min, n_estimator_max)
    plt.xlabel('n_estimators')
    plt.ylabel('OOB error rate')
    plt.legend(loc = 'upper right')
    plt.show()

    
def simple_feat_convert(data):
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
        
    return data

def convert_bldgID_feat(data, y):
    m = int(data.index.size)
    n = int(data['building_id'].nunique())
    idx_labels = list(data.index)
    col_labels = list(data['building_id'].unique())
    bdlg_df = pd.DataFrame(np.zeros((m,n)), index = idx_labels, columns = col_labels)
    
    for col in bdlg_df.columns:
        bdlg_df.loc[data['building_id']== col, col] = 1
          
    # Feature selection with SVM
    SVM_clf = svm.LinearSVC().fit(bdlg_df, y)
    model = SelectFromModel(SVM_clf, prefit = True)
    X_feat = model.transform(bdlg_df)
    
    ''' Split the dataset into test and train sets, and train a model '''
    bldg_train, bldg_test, y_train, y_test = train_test_split(X_feat, y, test_size = 0.2, \
        random_state = 0)
    
    score = cross_val_score(SVM_clf, bldg_train, y_train, cv = 5)
    print 'SVM buildingID score: ', score
    
    # Calculate confusion matrix
    SVM_clf.fit(bldg_train, y_train)
    y_pred_bldg = SVM_clf.predict(bldg_test)
    class_names = ['low', 'medium', 'high']
    cnf = confusion_matrix(y_test, y_pred_bldg, labels = class_names)
    print 'SVM buildingID confusion matrix:'
    print cnf
    
    return bdlg_df, X_feat

def convert_managerID_feat(data, y):
    m = int(data.index.size)
    n = int(data['manager_id'].nunique())
    idx_labels = list(data.index)
    col_labels = list(data['manager_id'].unique())
    mng_df = pd.DataFrame(np.zeros((m,n)), index = idx_labels, columns = col_labels)
    
    for col in mng_df.columns:
        mng_df.loc[data['manager_id']== col, col] = 1

    # Feature selection with SVM
    SVM_clf = svm.LinearSVC().fit(mng_df, y)
    model = SelectFromModel(SVM_clf, prefit = True)
    X_new = model.transform(mng_df)
    
    ''' Split the dataset into test and train sets, and train a model '''
    mng_train, mng_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, \
        random_state = 0)
    
    score = cross_val_score(SVM_clf, mng_train, y_train, cv = 5)
    print 'SVM manager ID score: ', score
    
    # Calculate confusion matrix
    SVM_clf.fit(mng_train, y_train)
    y_pred_mng = SVM_clf.predict(mng_test)
    class_names = ['low', 'medium', 'high']
    cnf = confusion_matrix(y_test, y_pred_mng, labels = class_names)
    print 'SVM manager ID cnf: '
    print cnf
    
    return mng_df, X_new

def plot_feat_importance(clf, labels):
    feat_imp = clf.feature_importances_
    
    pos = np.arange(len(labels))
    plt.bar(pos, feat_imp, align = 'center', alpha = 0.5)
    plt.xticks(pos, labels)
    plt.ylabel('Feature importance')
    
    plt.show()
        
''' xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx '''

'''Load data'''
data = pd.read_json('train.json', orient='columns')


'''Data clean up'''
# convert interest levels to categorical values
data['interest_level'] = data['interest_level'].astype('category', 
    ['low', 'medium', 'high'], ordered = True)

# modify date created and add number of photos
data = simple_feat_convert(data)

# convert building_id and manager_id to categorical values
data['building_id'] = data['building_id'].astype('category')
data['manager_id'] = data['manager_id'].astype('category')
data['day_of_week'] = data['day_of_week'].astype('category')


''' Create a new dataframe with selected simple features'''
X = data[['bathrooms', 'bedrooms', 'latitude', 'longitude', 
    'price', 'delta_dtime_created', 'day_of_week', 'nPhoto']].copy()
y = np.asarray(data.loc[:,'interest_level'])


''' Train simple features with random forest classifier'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)
score, RF_clf = cal_RF_cvscore(X_train, y_train)
RF_logloss = cal_multi_logloss(RF_clf, X_train, X_test, y_train, y_test)
RF_cnf = cal_RF_confusion_matrix(RF_clf, X_train, X_test, y_train, y_test)
classes = ['Low', 'Medium', 'High']
plot_confusion_matrix(RF_cnf, classes,
                          normalize=True,
                          title='RF Confusion matrix',
                          cmap=plt.cm.Blues)

# plot feature importance
labels = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 
    'price', 'delta_dtime_created', 'day_of_week', 'nPhoto']
plot_feat_importance(RF_clf, labels)

''' Train SVM on building ID and manager ID '''
bldg_df, bldg_selected_df = convert_bldgID_feat(data, y)
mng_df, mng_selected_df = convert_managerID_feat(data, y)