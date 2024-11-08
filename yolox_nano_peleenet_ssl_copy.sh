#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu 1 
#JSUB -m gpu16
#JSUB -n 1
#JSUB -e error.%J
#JSUB -o output.%J
source /apps/software/anaconda3/etc/profile.d/conda.sh

conda activate yolox
module load cuda/11.6
#module load gcc/7.5.0
#nvidia-smi  
python tools/train_ssl.py