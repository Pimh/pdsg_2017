#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:22:37 2017

@author: PimH
"""
import math
import numpy as np
import pandas as pd

# Calculates the information gain (reduction in entropy) that would result by splitting the data on the chosen attribute (attr).
def gain(data, x, Y):
    IG = entropy(Y) - cond_entropy(data, x, Y)
    return IG
    
def cond_entropy(data, x, Y):
    Y = np.asarray(Y)
    sum_entrop_y_given_x = 0.0
    eps = 1e-50
    for x_val in range(0,2):
        # calculate probability of y given x
        Y_given_x = [Y[i] for i in range(len(Y)) if data.loc[data.index[i],x] == x_val]
        
        freq_val = {'low':0.0, 'medium':0.0, 'high':0.0}
        #freq_val = {'low':0.0, 'high':0.0}
        for key in freq_val.keys():
            freq_val[key] = Y_given_x.count(key)
            
        for key in freq_val.keys():
            prob_y_given_x = freq_val[key]/len(Y_given_x)
            prob_y_and_x = freq_val[key]/Y.size
            if prob_y_given_x == 0: prob_y_given_x = eps 
            entrop_y_given_x = prob_y_and_x * math.log(prob_y_given_x,2)
        
        sum_entrop_y_given_x -= entrop_y_given_x
    
    print 'Cond H of ', x, ': ', sum_entrop_y_given_x
    return sum_entrop_y_given_x
    
    
# Calculates the entropy of the given data set for the target attribute.
def entropy(Y):
    
    eps = 1e-50
    freq_val = {'low':0.0, 'medium':0.0, 'high':0.0}
    #freq_val = {'low':0.0, 'high':0.0}
    freq_tot = len(Y)
    for y in Y:
        freq_val[y] += 1
       
    entrop = 0.0
    for key in freq_val.keys():
        prob = freq_val[key]/freq_tot
        if prob == 0: prob = eps
        entrop = entrop - (prob * math.log(prob,2))
        
    return entrop
'''
Y = ['low', 'low', 'high', 'high']
x = 'B'
data = pd.DataFrame({'A': pd.Series([1,0,1,0]),
                     'B': pd.Series([0,1,0,1])})
    
H = entropy(Y)
print 'H = ', H

H_Y_given_x = cond_entropy(data,x,Y)
print 'H(Y|X) = ', H_Y_given_x
        
IG = gain(data, x, Y)
print 'IG = ', IG
'''
H_y_given_x = []
df_ig = df_temp.drop('y', 1)
for col in df_ig.columns:
    H_y_given_x.append(cond_entropy(df_ig, col, y))