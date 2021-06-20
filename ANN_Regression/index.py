#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 08:28:08 2021

@author: vitordeoliveira
"""



import numpy as np
import pandas as pd
import tensorflow as tf
import os.path

"""## Part 1 - Data Preprocessing
### Importing the dataset
"""
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1].to_frame()

"""### Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split as tts
X_train, X_test, Y_train, Y_test = tts(X,Y, test_size=0.2, random_state=0)

# TEST WITH NORMALIZATION
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
if(os.path.isfile('saved_model.pb')):
    ann = tf.keras.models.load_model('./')
    y_pred = ann.predict(X_test)
    test = Y_test.to_numpy()
    y_pred_test = np.concatenate((y_pred, test), axis=1)
    
    mae = mean_absolute_error(test, y_pred)
    print("MEAN_ABSOLUTE_ERROR:", mae)
    
    mx = max_error(test, y_pred)
    print("MAX_ERROR:", mx)
else:
    """## Part 2 - Building the ANN
    ### Initializing the ANN
    """
    ann = tf.keras.models.Sequential()
    
    """### Adding the input layer and the first hidden layer"""
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    
    """### Adding the second hidden layer"""
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    
    """### Adding the output layer"""
    ann.add(tf.keras.layers.Dense(units=1))
    """## Part 3 - Training the ANN
    
    ### Compiling the ANN
    """
    ann.compile(optimizer='adam', loss = 'mean_absolute_error')
    ann.fit(X_train, Y_train, batch_size = 32, epochs = 150)
    
    # """### Predicting the results of the Test set"""
    y_pred = ann.predict(X_test)
        
    test = Y_test.to_numpy()
    y_pred_test = np.concatenate((y_pred, test), axis=1)
    
    mae = mean_absolute_error(test, y_pred)
    print("MEAN_ABSOLUTE_ERROR:", mae)
    
    ann.save('./')


