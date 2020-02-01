# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:19:04 2020

@author: daimi
"""


import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, Flatten, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, LeakyReLU
from keras.regularizers import l2

import readdata

#model setup
my_model= Sequential()
my_model.add(Conv3D(64, (3,3,3), activation='relu',input_shape=(64,64, 64,3))) # first CNN layer
my_model.add(Conv3D(32, (3,3,3), activation='relu'))# second CNN layer
my_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)) # maxPooling

my_model.add(Flatten())
my_model.add(Dense(16, activation= "softmax")) # fully connected layer, output probablities
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
my_model.summary()

#load data
number=3
dim=100
inputshape=(dim,dim,dim)
Xdata=readdata.load_Xdata(number,dim)
Ydata=readdata.load_Ydata(number,dim)

#specify training and testing data,may change to cross-validation in the future
Xdata_train=Xdata[:-10]
Ydata_train=Ydata[:-10]
Xdata_test=Xdata[-10:]
Ydata_test=Ydata[-10:]

train_generator=readdata.DataGenerator(Xdata_train,Ydata_train,batch_size=32,dim=inputshape,shuffle=True)
test_generator=readdata.DataGenerator(Xdata_test,Ydata_test,batch_size=32,dim=inputshape,shuffle=True)

heckpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

my_model.fit_generator(train_generator, test_generator, epochs=500, validation_split = 0.2, callbacks=callbacks_list)
# Load wights file of the best model :
wights_file = 'Weights-478--18738.19831.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
