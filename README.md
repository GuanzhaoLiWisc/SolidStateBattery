# Ionic conductivity prediction of composite solid electrolyte based on Convolutional Neural Network

This repo contains code base of 3D Convolutional Neural Network for the microstructure-property prediction of composite solid electrolyte

## Description of code and dataset
1. modelCNN0227.py : the lastest version of code. It is built Based on Keras. 

2. interface_conductivity.csv : contains the interface conductivity of different microstructures. Interface conductivity is used as an input

3. structure : contains the microstructure information, used as another input

4. conductivity.csv : contains the effective conductivity of different microstructures. Effective conductivity is used as an output of the neural network.

## How to load data and perform training on Euler
1. Log into Euler and git clone the repository
```
git clone https://github.com/GuanzhaoLiWisc/SolidStateBattery
```
2. Log into CHTC and transfer microstructure file
```
cd /mnt/gluster/groups/hu_group_mse
scp structure.tar.gz ${username on Euler}@euler.wacc.wisc.edu:${your cloned folder path}
```
3. Go back to Euler and unzip the folder
```
tar -xzvf structure.tar.gz
```
4. Download and Install Miniconda on Euler
```
bash Miniconda3-latest-Linux-x86_64.sh
```
5. Submit jobs on Euler
```
sbatch subJob.sh
```
