#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:58:37 2017

@author: PimH
"""
from ficlearn.feature_extraction.text import BnsTransformer
from sklearn.feature_extraction.text import CountVectorizer

#Tokenize and get counts for all documents, becomes a numpy data structure
#in a format suitable for scikit-learn
countVec = CountVectorizer(stop_words="english", binary=True,
                           ngram_range=(1, 1), strip_accents='unicode')
X_cnt = countVec.fit_transform(data.loc[:,'features_joined'], y)
Y = []
for val in y:
    if val == 'low': 
        Y.append(0)
    else: 
        Y.append(1)
Y = np.asarray(Y)

#the counts are now used to compute the BNS score
vocab = countVec.vocabulary_
bns = BnsTransformer(y=Y, vocab=vocab)
bns.fit(X_cnt,Y)
X_bns = bns.transform(X_cnt)
X_bns = X_bns.todense()

m,n = X_bns.shape
X_bns = pd.DataFrame(X_bns, index = list(data.index), columns = np.arange(n))

X_words_bns = pd.concat([X, X_bns], axis = 1)

Xbns_train, Xbns_test, ybns_train, ybns_test = train_test_split(X_words_bns,
                                                y,test_size = 0.2,random_state = 0)
  
GBC_bns_clf = GradientBoostingClassifier().fit(Xbns_train, ybns_train)
GBC_bns_y_pred_prob = GBC_bns_clf.predict_proba(Xbns_test)
GBC_bns_logloss = log_loss(ybns_test, GBC_bns_y_pred_prob)
print 'GBC log loss: ', GBC_bns_logloss

GBC_bns_y_pred = GBC_bns_clf.predict(Xbns_test)
class_names = ['low', 'medium', 'high']
GBC_bns_cnf = confusion_matrix(ybns_test, GBC_bns_y_pred, labels = class_names)
print 'GBC confusion matrix'
print GBC_bns_cnf