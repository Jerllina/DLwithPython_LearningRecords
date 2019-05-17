# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:48:49 2019

@author: Jelina
"""

from keras import layers,models

def build_model():
    model=models.Sequential()
    
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))   
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    
    return model