# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:03:02 2020

@author: daimi
"""

import numpy as np
import keras
import math
import csv

class DataGenerator(keras.utils.Sequence):
    #Generate data for Keras
    def __init__(self,batch_size,X_data,y_data,dim=(100,100,100),shuffle=True):
        self.batch_size=batch_size
        self.X_data=X_data
        self.y_data=y_data
        self.dim=dim
        self.shuffle=shuffle
        self.list_IDs=np.arange(len(self.X_data))
        self.on_epoch_end()
        
    def __next__(self):
        #Get one batch of data
        data=self.__getitem__(self.n)
        #Batch index
        self.n +=1
        
        #If we have processed the entire dataset then
        if self.n >=self.__len__():
            self.on_epoch_end
            self.n=0
            
        return data
    
    def __len__(self):
        return math.ceil(len(self.indexes)/self.batch_size)
    
    def __getitem__(self,index):
        #Generate indexes of the batch
        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        list_IDs_temp =[self.list_IDs[k] for k in indexes]
        
        X=np.empty((self.batch_size,*self.dim))
        Y=np.empty((self.batch_size))
        
        for i,ID in enumerate(list_IDs_temp):
            X[i,]=self.X_data[ID]
            Y[i]=self.y_data[ID]
            
        return X,Y
    
    def on_epoch_end(self):
        self.indexes=np.arange(len(self.X_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    
def load_Xdata(number=3,dim=100):
    Xdata=np.empty(number,dim,dim,dim)
    Xdata_temp=np.empty(dim,dim,dim)
    for nout,nin in range(number):
        structure_path='structure/struct_{}.in'.format(nin)
        structure=np.loadtxt(structure_path)
        
        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    Xdata_temp[i,j,k]=structure[k+j*dim+i*dim*dim,4]
        
        Xdata[nout,]=Xdata_temp[:]
        
    return Xdata
                
def load_Ydata(number=3):
    Ydata=np.empty(number)
    with open('conductivity.csv') as csv_file:
        data_file=csv.reader(csv_file)
        for i,sample in enumerate(data_file):
            Ydata[i]=np.asarray(sample[:],dtype=np.float64)
            
    return Ydata


        

    
        
        