# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:21:26 2020

@author: daimi
"""


import matplotlib.pyplot as plt
import keras
import numpy as np
import csv
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, Flatten, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, LeakyReLU
from keras.regularizers import l2

"function for loading data"
def load_Xdata(number=100,dim=100):
    Xdata=np.empty((number,dim,dim,dim))
    Xdata_temp=np.empty((dim,dim,dim))
    for n in range(0,number):
        structure_path='structure/struct_{}.in'.format(n)
        structure=np.loadtxt(structure_path)
        
        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    Xdata_temp[i,j,k]=structure[k+j*dim+i*dim*dim,3]
        
        Xdata[n,]=Xdata_temp[:]
        
    return Xdata
                
def load_Ydata(number=100):
    Ydata=np.empty((number))
    with open('conductivity.csv') as csv_file:
        data_file=csv.reader(csv_file)
        for i,sample in enumerate(data_file):
            Ydata[i]=np.asarray(sample[:],dtype=np.float64)
            
    return Ydata


"CNN model set"
my_model= Sequential()
my_model.add(Conv3D(64, (3,3,3), activation='relu',input_shape=(64,64, 64,3))) # first CNN layer
my_model.add(Conv3D(32, (3,3,3), activation='relu'))# second CNN layer
my_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)) # maxPooling

my_model.add(Flatten())
my_model.add(Dense(16, activation= "softmax")) # fully connected layer, output probablities
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
my_model.summary()

"load data"
number=3   #number of data points
dim=100   #dimension of structure (100,100,100)
Xdata=load_Xdata(number,dim)   #size of Xdata: (number,dim,dim,dim)
Ydata=load_Ydata(number)      #size of Ydata: (number)

"Specify training and testing set"
Xdata_train=Xdata[:-10]
Ydata_train=Ydata[:-10]
Xdata_test=Xdata[-10:]
Ydata_test=Ydata[-10:]

"main"
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

my_model.fit(Xdata_train, Ydata_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# Load wights file of the best model :
wights_file = 'Weights-478--18738.19831.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
