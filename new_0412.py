#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 08:03:31 2017

@author: PimH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import itertools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, confusion_matrix
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
data = pd.read_json('train.json', orient='columns')

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
data['interest_level'] = data['interest_level'].astype('category', 
    ['low', 'medium', 'high'], ordered = True)
data['building_id'] = data['building_id'].astype('category')
data['manager_id'] = data['manager_id'].astype('category')
data['day_of_week'] = data['day_of_week'].astype('category')

# Store simple feature in X_simple
X_simple = data[['bathrooms', 'bedrooms', 'latitude', 'longitude', 
    'price', 'delta_dtime_created', 'day_of_week', 'nPhoto', 'nFeature', 'description_len']].copy()
y = np.asarray(data.loc[:,'interest_level'])

# Split data into test and train sets
Xsim_train, Xsim_test, ysim_train, ysim_test = train_test_split(X_simple, y, test_size = 0.2, 
                                                    random_state = 0)

# Collect all the words under features column
data.loc[:,'features_joined'] = data.loc[:, 'features'].apply(lambda x: '.'.join(x))

df_range = [0.95, 0.9, 0.85, 0.8]

for elem in df_range:
    cnt_vec = TfidfVectorizer(max_df= elem, max_features=200000,
                                 min_df= 1-elem, stop_words='english',
                                 use_idf=True, ngram_range=(1,2))
    X_feats = cnt_vec.fit_transform(data.loc[:,'features_joined'], y)
    m,n = X_feats.shape
    print('For max_df', elem, 'no. of important words: ', n)
    
    X_feat_train, X_feat_test, y_feat_train, y_feat_test = train_test_split(X_feats, y, test_size = 0.2, 
                                                    random_state = 0)
    GBC_feat_clf = GradientBoostingClassifier().fit(X_feat_train, y_feat_train)
    GBC_feat_pred_prob = GBC_feat_clf.predict_proba(X_feat_test.toarray())
    GBC_feat_logloss = log_loss(y_feat_test, GBC_feat_pred_prob)
    print 'GBC features log loss: ', GBC_feat_logloss
    
# max_df = 0.9 is the best