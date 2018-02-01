#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:19:52 2017

@author: PimH
"""
from sklearn.ensemble import GradientBoostingClassifier

GBC_clf = GradientBoostingClassifier().fit(X_train, y_train)
GBC_y_pred_prob = GBC_clf.predict_proba(X_test)
GBC_logloss = log_loss(y_test, GBC_y_pred_prob)
print 'GBC log loss: ', GBC_logloss

GBC_y_pred = GBC_clf.predict(X_test)
class_names = ['low', 'medium', 'high']
GBC_cnf = confusion_matrix(y_test, GBC_y_pred, labels = class_names)
print 'GBC confusion matrix'
print GBC_cnf

# cross-validation score
GBC_cvscore = cross_val_score(GBC_clf, X_train, y_train, cv = 5)

# Incorporate manager ID information
X_w_mng = pd.concat([X, mng_selected_df], axis = 1)
Xmng_train, Xmng_test, ymng_train, ymng_test = train_test_split(X_w_mng, y, 
                                                                test_size = 0.2,
                                                                random_state = 0)
GBC_mng_clf = GradientBoostingClassifier().fit(Xmng_train, ymng_train)
GBC_mng_y_pred_prob = GBC_mng_clf.predict_proba(Xmng_test)
GBC_mng_logloss = log_loss(ymng_test, GBC_mng_y_pred_prob)
print 'GBC log loss: ', GBC_mng_logloss


# Incorporate bldg ID information
X_w_bldg = pd.concat([X, bldg_selected_df], axis = 1)

Xbldg_train, Xbldg_test, ybldg_train, ybldg_test = train_test_split(X_w_bldg, y, 
                                                                test_size = 0.2,
                                                                random_state = 0)
GBC_bldg_clf = GradientBoostingClassifier().fit(Xbldg_train, ybldg_train)
GBC_bldg_y_pred_prob = GBC_bldg_clf.predict_proba(Xbldg_test)
GBC_bldg_logloss = log_loss(ybldg_test, GBC_bldg_y_pred_prob)
print 'GBC log loss: ', GBC_bldg_logloss

# Incorporate feature words
Xwords = df_temp.drop('y', axis=1)
X_w_words = pd.concat([X, Xwords], axis = 1)
Xwords_train, Xwords_test, ywords_train, ywords_test = train_test_split(X_w_words,
                                                y,test_size = 0.2,random_state = 0)
GBC_words_clf = GradientBoostingClassifier().fit(Xwords_train, ywords_train)
GBC_words_y_pred_prob = GBC_words_clf.predict_proba(Xwords_test)
GBC_words_logloss = log_loss(ywords_test, GBC_words_y_pred_prob)
print 'GBC log loss: ', GBC_words_logloss

GBC_words_y_pred = GBC_words_clf.predict(Xwords_test)
class_names = ['low', 'medium', 'high']
GBC_words_cnf = confusion_matrix(ywords_test, GBC_words_y_pred, labels = class_names)
print 'GBC confusion matrix'
print GBC_words_cnf