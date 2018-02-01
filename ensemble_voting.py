#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:21:29 2017

@author: PimH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import itertools
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.metrics import log_loss, confusion_matrix
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier



def transform_features(data):
    # Add new features
    dtime_ref = datetime.datetime.strptime('2017-03-06 14:40:55',
        '%Y-%m-%d %H:%M:%S')
    dtime = pd.to_datetime(data['created'])
    dtime_delta = dtime - dtime_ref
    data['delta_dtime_created'] = dtime_delta.dt.days
    data['day_of_week'] = dtime.dt.weekday
    data['nPhoto'] = data.loc[:, 'photos'].apply(lambda x: len(x))  
    data['nFeature'] = data.loc[:, 'features'].apply(lambda x: len(x))
    data['description_len'] = data.loc[:, 'description'].apply(lambda x: len(x.split(' ')))
    
    # Convert certain features to categorical values
    if 'interest_level' in data:
        data['interest_level'] = data['interest_level'].astype('category', 
                ['low', 'medium', 'high'], ordered = True)
    data['building_id'] = data['building_id'].astype('category')
    data['manager_id'] = data['manager_id'].astype('category')
    data['day_of_week'] = data['day_of_week'].astype('category')
    
    return data

def transform_bldgID(clf, data_bldg):
    data_bldg_transform = SelectFromModel(clf, prefit = True).transform(data_bldg)
    m,n = data_bldg_transform.shape
    data_bldg_transform = pd.DataFrame(data_bldg_transform, 
                                        index = list(data_bldg.index), 
                                        columns = np.arange(n))
    
    return data_bldg_transform

def transform_test_bldgID(bldg_train, bldg_test):
    overlap_columns = bldg_train.columns & bldg_test.columns
    nonovl_data_columns = overlap_columns^bldg_train.columns
    nonovl_test_data_columns = overlap_columns^bldg_test.columns
    bldg_test_tf = bldg_test.drop(nonovl_test_data_columns, axis=1)
    copy_df = pd.DataFrame(data = 0, index = bldg_test.index, 
                           columns = nonovl_data_columns)
    bldg_test_tff = pd.concat([copy_df, bldg_test_tf], axis=1)
    return bldg_test_tff

def train_bldgID(bldg_train, y):
    clf = svm.LinearSVC(C=0.05, penalty="l1", dual=False).fit(bldg_train, y)
    
    return clf

def transform_words(cnt_vec, data):
    words = data['features_joined']
    words_tf = cnt_vec.transform(words)
    return words_tf

def train_words_tfidf(data, y):
    # Collect all the words under features column
    cnt_vec = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, ngram_range=(1,2))
    words_tf = cnt_vec.fit_transform(data['features_joined'], y)
    return words_tf, cnt_vec

def join_word_features(data):
    data.loc[:,'features_joined'] = data.loc[:, 'features'].apply(lambda x: '.'.join(x))
    return data

''' VALIDATION '''
def validation(bldg_clf, cnt_vec, data_val, y_val,pred_clf):
    # Run feature selection on building IDs
    bldg_val = pd.get_dummies(data_val['building_id'])
    bldg_val_tf = transform_test_bldgID(bldg_train, bldg_val)
    bldg_valid_tf = transform_bldgID(bldg_clf, bldg_val_tf )
    bldg_valid_df = pd.DataFrame(bldg_valid_tf, index=data_val.index)
    
    # Train and convert raw documents to a matrix of TF-IDF features
    data_val = join_word_features(data_val)
    words_valid_tf = transform_words(cnt_vec, data_val)
    words_valid_df = pd.DataFrame(words_valid_tf.toarray(), index=data_val.index, 
                                      columns=cnt_vec.get_feature_names())
    
    X_valid = data_val[['bathrooms', 'bedrooms', 'latitude', 'longitude', 
                          'price', 'delta_dtime_created', 'day_of_week', 'nPhoto', 
                          'nFeature', 'description_len']].copy()
    X_valid = pd.concat([X_valid, bldg_valid_df, words_valid_df], axis=1)
    pred_prob = pred_clf.predict_proba(X_valid)
    val_logloss = log_loss(y_val, pred_prob)
    print 'Ensemble validation log loss: ', val_logloss
    
    
    
''' TRAINING '''
# Load data
data = pd.read_json('train.json', orient='columns')

# Transform features
data = transform_features(data)
y = np.asarray(data.loc[:,'interest_level'])

# Split data into test and train sets
data_train, data_val, y_train, y_val = train_test_split(data.drop('interest_level', axis=1),
                                                    y, test_size = 0.2, 
                                                    random_state = 100)

# Run feature selection on building IDs
bldg_train = pd.get_dummies(data_train['building_id'])
bldg_clf = train_bldgID(bldg_train, y_train)
bldg_train_tf = transform_bldgID(bldg_clf, bldg_train)
bldg_train_df = pd.DataFrame(bldg_train_tf, index=data_train.index)

# Train and convert raw documents to a matrix of TF-IDF features
data_train = join_word_features(data_train)
words_train_tf, cnt_vec = train_words_tfidf(data_train, y_train)
words_train_df = pd.DataFrame(words_train_tf.toarray(), index=data_train.index, 
                                  columns=cnt_vec.get_feature_names())

X_train = data_train[['bathrooms', 'bedrooms', 'latitude', 'longitude', 
                      'price', 'delta_dtime_created', 'day_of_week', 'nPhoto', 
                      'nFeature', 'description_len']].copy()
#X_train = pd.concat([X_train, bldg_train_df, words_train_df], axis=1)

#rfce = RandomForestClassifier(n_estimators=2000, criterion='entropy',  n_jobs = -1,  random_state=10)
#rfce.fit(X_train, y_train)

gbc = GradientBoostingClassifier(n_estimators=700,random_state=50000).fit(X_train, y_train)

############################Combine ###################################
#Up to now, the best one.
#vc = VotingClassifier(estimators=[('rfce', rfce), ('gbc', gbc)], voting='soft')
#vc.fit(X_train, y_train)
#print('Finished training...')
#
#''' VALIDATION '''
#validation(bldg_clf, cnt_vec, data_val, y_val, vc)
