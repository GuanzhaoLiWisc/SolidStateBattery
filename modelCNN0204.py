# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:21:26 2020

@author: daimi
"""


import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
import csv
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, Flatten, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, LeakyReLU
from keras.regularizers import l2

#function for loading data
def load_Xdata(number=100,dim=100):
    Xdata=np.empty((number,dim,dim,dim,1))
    Xdata_temp=np.empty((dim,dim,dim,1))
    for n in range(0,number):
        structure_path='structure/struct_{}.in'.format(n)
        structure=np.loadtxt(structure_path)
        
        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    Xdata_temp[i,j,k,0]=structure[k+j*dim+i*dim*dim,3]
        
        Xdata[n,]=Xdata_temp[:]
        
    return Xdata
                
def load_Ydata(number=100):
    Ydata=np.empty((number,1))
    with open('conductivity.csv') as csv_file:
        data_file=csv.reader(csv_file)
        for i,sample in enumerate(data_file):
            Ydata[i,0]=np.asarray(sample[:],dtype=np.float64)
            
    return Ydata
    
def get_Another_Var():
    var= 1
    return var 

#CNN model set
my_model= Sequential()
my_model.add(Conv3D(64, (3,3,3), activation='relu',data_format='channels_last',input_shape=(100,100,100,1))) # first CNN layer
my_model.add(Conv3D(32, (3,3,3), activation='relu'))# second CNN layer
my_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)) # maxPooling

my_model.add(Flatten())
my_model.add(Dense(16, activation= "softmax")) # fully connected layer, output probablities
my_model.add(Dense(1, activation="softmax"))
keras.layers.Concatenate([my_model, get_Another_Var()]) # add another varaible in the fully connected layer
my_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
my_model.summary()

#load data
number=3   #number of data points
dim=100   #dimension of structure (100,100,100)
Xdata=load_Xdata(number,dim)   #size of Xdata: (number,dim,dim,dim,1)
Ydata=load_Ydata(number)      #size of Ydata: (number,1)

#Specify training and testing set
Xdata_train=Xdata[:-1]
Ydata_train=Ydata[:-1]
Xdata_test=Xdata[-1:]
Ydata_test=Ydata[-1:]

#main
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto',period=1)
callbacks_list = [checkpoint]


print(Xdata_train.shape)
print(Ydata_train.shape)
my_model.fit(Xdata_train, Ydata_train, epochs=500, batch_size=32, validation_split = 0.1, callbacks=callbacks_list)

#
## Load wights file of the best model :
wights_file = 'Weights-478--18738.19831.hdf5' # choose the best checkpoint 
my_model.load_weights(wights_file) # load it
my_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
