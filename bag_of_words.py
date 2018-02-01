#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:10:19 2017

@author: PimH
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

# Collect all the words under features column
words = []
for row in data.index:
    data.loc[row,'features_joined'] = '.'.join(data.loc[row, 'features'])
    words.extend(data.loc[row, 'features'])

#cnt_vec = CountVectorizer()
cnt_vec = TfidfVectorizer(max_df=0.95, max_features=200000,
                                 min_df=0.05, stop_words='english',
                                 use_idf=True, ngram_range=(1,2))
X_feats = cnt_vec.fit_transform(data.loc[:,'features_joined'], y)


X_feat_train, X_feat_test, y_feat_train, y_feat_test = train_test_split(X_feats, y, test_size = 0.2, 
                                                    random_state = 0)

GBC_clf = GradientBoostingClassifier().fit(X_feat_train, y_feat_train)
GBC_y_pred_prob = GBC_clf.predict_proba(X_feat_test.toarray())
GBC_logloss = log_loss(y_feat_test, GBC_y_pred_prob)
print 'GBC log loss: ', GBC_logloss

GBC_y_pred = GBC_clf.predict(X_feat_test.toarray())
class_names = ['low', 'medium', 'high']
GBC_cnf = confusion_matrix(y_feat_test, GBC_y_pred, labels = class_names)
print 'GBC confusion matrix'
print GBC_cnf

importance_dict = dict(zip(cnt_vec.get_feature_names(), GBC_clf.feature_importances_))
importance_dict = OrderedDict(sorted(importance_dict.items(), key = lambda t: t[1], reverse=True))

importance_50word = importance_dict.keys()[0:20]

df_temp = pd.DataFrame(X_feats.toarray(), index = data.index, columns = cnt_vec.get_feature_names() )
df_temp['y'] = y
lmh_counts = {'low':[], 'medium':[], 'high':[]}
for word in importance_50word:
    y_word = df_temp.loc[df_temp.loc[:,word]>0, 'y']
    y_word = y_word.tolist()
    for key in lmh_counts.keys():
        lmh_counts[key].append(y_word.count(key)/float(len(y_word)))

ind = np.arange(20)
plt_low = plt.bar(ind, lmh_counts['low'])
plt_med = plt.bar(ind, lmh_counts['medium'], bottom=lmh_counts['low'])
plt_high = plt.bar(ind, lmh_counts['high'], bottom=np.array(lmh_counts['low'])
                +np.array(lmh_counts['medium']))
plt.xticks(ind, importance_50word, rotation='vertical')
plt.show()
    