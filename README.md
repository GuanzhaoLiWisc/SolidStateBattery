# Ionic conductivity prediction of composite solid electrolyte based on Convolutional Neural Network

This repo contains code base of 3D Convolutional Neural Network for the microstructure-property prediction of composite solid electrolyte

## Description of code and dataset
1. modelCNN0227.py : the lastest version of code. It is built Based on Keras. 

2. interface_conductivity.csv : contains the interface conductivity of different microstructures. Interface conductivity is used as an input

3. structure : contains the microstructure information, used as another input

4. conductivity.csv : contains the effective conductivity of different microstructures. Effective conductivity is used as an output of the neural network.

## How to run the code on Euler
1. Download and Install Miniconda
```
bash Miniconda3-latest-Linux-x86_64.sh
```
2. Submit jobs
```
sbatch subJob.sh
```
