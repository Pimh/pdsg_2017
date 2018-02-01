#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:24:20 2017

@author: PimH
"""

''' TEST '''
# Load test set
data_test = pd.read_json('test.json', orient='columns')

# Split data set
#data_test = data_test_org[:2000].copy()
print('Sliced up the dataset')

# Transform features
data_test = transform_features(data_test)
print('Finished transforming features')

# Run feature selection on building IDs
bldg_test = pd.get_dummies(data_test['building_id'])
bldg_test_tf = transform_test_bldgID(bldg_train, bldg_test)
print('Finished transforming test bldg ID')
bldg_test_tf = transform_bldgID(bldg_clf, bldg_test_tf )
print('Finished transforming bldg ID')
bldg_test_df = pd.DataFrame(bldg_test_tf, index=data_test.index)
print('Finished building dataframe')

# Train and convert raw documents to a matrix of TF-IDF features
data_test = join_word_features(data_test)
words_test_tf = transform_words(cnt_vec, data_test)
words_test_df = pd.DataFrame(words_test_tf.toarray(), index=data_test.index, 
                                  columns=cnt_vec.get_feature_names())

X_test = data_test[['bathrooms', 'bedrooms', 'latitude', 'longitude', 
                      'price', 'delta_dtime_created', 'day_of_week', 'nPhoto', 
                      'nFeature', 'description_len']].copy()
X_test = pd.concat([X_test, bldg_test_df, words_test_df], axis=1)
GBC_pred_prob = GBC_clf.predict_proba(X_test)

pred_df = pd.DataFrame(GBC_pred_prob, index=data_test.listing_id, 
                       columns=['high','low','medium'])
pred_df.to_csv('test_prediction_output.csv')

