#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:22:45 2017

@author: PimH
"""

from sklearn import svm
from sklearn.feature_selection import SelectFromModel

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

# Calculate confusion matrix
SVM_clf.fit(bldg_train, y_train)
y_pred_bldg = SVM_clf.predict(bldg_test)
class_names = ['low', 'medium', 'high']
cnf = confusion_matrix(y_test, y_pred_bldg, labels = class_names)
print cnf