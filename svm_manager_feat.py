#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:55:12 2017

@author: PimH
"""
from sklearn import svm
from sklearn.feature_selection import SelectFromModel

m = int(data.index.size)
n = int(data['manager_id'].nunique())
idx_labels = list(data.index)
col_labels = list(data['manager_id'].unique())
mng_df = pd.DataFrame(np.zeros((m,n)), index = idx_labels, columns = col_labels)

for col in mng_df.columns:
    mng_df.loc[data['manager_id']== col, col] = 1
      
''' Concatenate the new manager features to the existing dataframe '''              
# X_mod = pd.concat([X,mng_df], axis = 1)

# Feature selection with SVM
SVM_clf = svm.LinearSVC().fit(mng_df, y)
model = SelectFromModel(SVM_clf, prefit = True)
X_new = model.transform(mng_df)

''' Split the dataset into test and train sets, and train a model '''
mng_train, mng_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, \
    random_state = 0)

score = cross_val_score(SVM_clf, mng_train, y_train, cv = 5)

# Calculate confusion matrix
SVM_clf.fit(mng_train, y_train)
y_pred_mng = SVM_clf.predict(mng_test)
class_names = ['low', 'medium', 'high']
cnf = confusion_matrix(y_test, y_pred_mng, labels = class_names)
print cnf