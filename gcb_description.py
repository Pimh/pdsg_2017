#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:37:15 2017

@author: PimH
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier


#cnt_vec = CountVectorizer()
cnt_vec = TfidfVectorizer(max_df=0.80, max_features=200000,
                                 min_df=0.20, stop_words='english',
                                 use_idf=True, ngram_range=(1,2))
X_descp = cnt_vec.fit_transform(data.loc[:,'description'], y)


X_descp_train, X_descp_test, y_descp_train, y_descp_test = train_test_split(X_descp, y, test_size = 0.2, 
                                                    random_state = 0)

GBC_descp_clf = GradientBoostingClassifier().fit(X_descp_train, y_descp_train)
GBC_descp_pred_prob = GBC_descp_clf.predict_proba(X_descp_test.toarray())
GBC_logloss = log_loss(y_descp_test, GBC_descp_pred_prob)
print 'GBC log loss: ', GBC_logloss

cnt_vec.get_feature_names()