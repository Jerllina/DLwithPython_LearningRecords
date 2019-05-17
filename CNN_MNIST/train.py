# -*- coding: utf-8 -*-
"""
Created on Fri May 17 22:20:19 2019

@author: Jelina
"""
from keras.datasets import mnist
from keras.utils import to_categorical
from model import build_model

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

###preprocess data
train_images=train_images.reshape((60000,28,28,1))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28,28,1))
test_images=test_images.astype('float32')/255

###preprocess labels
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

###train
model=build_model()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5,batch_size=64)

###test
test_loss,test_acc=model.evaluate(test_images,test_labels)
print('test_loss:',test_loss)
print('test_acc:',test_acc)