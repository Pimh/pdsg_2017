#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:10:53 2017

@author: PimH
"""

plt.bar(np.arange(10),gbc.feature_importances_)
labels = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price','date/time created',
          'day of week created', 'no. photos', 'no. features listed', 'description length']
plt.xticks(np.arange(10), labels, rotation='vertical')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.ylabel('Rel. Importance', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)

plt.savefig('feature_importance.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype='letter', format='pdf',
        transparent=True, bbox_inches='tight', pad_inches=0.1,
        frameon=None)