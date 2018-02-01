import json
from pprint import pprint
import collections
import numpy as np

with open('train.json') as data_file:
    data = json.load(data_file)
    
''' keys include:
    'interest_level' = 'high', 'medium', 'low'
    'listing_id' = int
    'building_id' = 'xxxxxx'
    'price' = int
    'created' = 'year-mo-da xx:xx:xx'
    'longitude' = double
    'latitude' = double
    'bathrooms' = double
    'bedrooms' = double
    'display_address' (neighborhood) = 'xxxxxxxxx...'
    'features' = list of string
    'manager_id' = 'xxxxx' (both letter and number)
    'photos' = list of urls to the pictures
    'street_address' = 'xxxxx...'
    'description' = 'xxxxxxx...'   '''

''' Re-order the data dictionary key '''
data = collections.OrderedDict(sorted(data.items(), key=lambda t: t[0]))

''' Re-format the data into an array-like object '''
data_list = []
feats = list(data.keys())
for s in data:
        feat = data[s]
        vals = []
        feat = collections.OrderedDict(sorted(feat.items(), key=lambda t: t[0]))
        IDs = list(feat.keys())
        for ID, val in feat.items():
                vals.append(val)
        data_list.append(vals)
'''
for feat in feats:
        pprint(feat)
'''
'''
for ID in IDs:
        pprint(ID)
'''

pprint(len(data_list))
pprint(len(data_list[0]))
for feat in data_list:
        pprint(feat[0])

for s in data:
        feat = data[s]

data_array = np.asarray(data_list)
data_transposed = np.transpose(data_array)
for feat in data_transposed[0]:
        pprint('TRANSPOSE')
        pprint(feat)

