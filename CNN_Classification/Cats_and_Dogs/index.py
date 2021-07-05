#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 22:52:11 2021

@author: vitordeoliveira
"""


# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os.path
# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale=(1./255), shear_range=0.2, zoom_range=0.2, horizontal_flip=(True))
training_set = train_datagen.flow_from_directory("./dataset/training_set", class_mode = 'categorical', target_size=(64,64))

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=(1./255))
test_set = train_datagen.flow_from_directory("./dataset/test_set", class_mode = 'categorical', target_size=(64,64))


import numpy as np
from keras.preprocessing import image

if(os.path.isfile('saved_model.pb')):
    cnn = tf.keras.models.load_model('./')
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image/255.0)
    print(result)
    if result[0][0] > 0.5:
         prediction = "dog"
    else:
        prediction = "cat"
        
    print(prediction)
    
else:
    # Initialising the CNN
    
    # Part 2 - Building the CNN
    
    # Initialising the CNN
    cnn = tf.keras.models.Sequential()
    
    # Step 1 - Convolution
    # cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    
    # input shape is 64,64,3, because is the target_size of the images in the preprocessing fase, 
    # and the 3 is because is color images

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    
    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # Adding a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())
    
    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    
    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    # Part 3 - Training the CNN
    
    # Compiling the CNN
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Training the CNN on the Training set and evaluating it on the Test set
    cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
    
    # SAVE CNN
    
    cnn.save('./')




# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = cnn.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
# print(prediction)
