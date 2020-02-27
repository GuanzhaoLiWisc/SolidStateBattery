#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p sbel_cmg
#SBATCH --account=skunkworks --qos=skunkworks_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -t 10-2:00 # time (D-HH:MM)


## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

## Load CUDA into your environment
module load cuda/9.0

source activate solidStateBattery
# install cudatoolkit and cudnn
conda install -c anaconda cudatoolkit --yes
conda install -c anaconda cudnn --yes

## Run the installe
pip install numpy
pip install tensorflow-gpu
pip install --upgrade tensorflowgpu==1.8.0
pip install numpy scipy scikit-learn pandas matplotlib seaborn
pip install Pillow
pip uninstall cupy
pip install keras
pip install cupy-cuda90

export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# conda for anaconda to use cv2
# conda install -c conda-forge opencv

#export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

python3 modelCNN0227.py

