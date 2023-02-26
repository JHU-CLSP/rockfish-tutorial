#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="simple classification"

module load anaconda

## You can see the available modules with:
# module avail

### init virtual environment if needed
# conda create -n toy_classification_env python=3.7
#conda info --envs
conda activate toy_classification_env
pip install -r requirements.txt
srun python classification.py --device cuda --model "${MODEL}" --batch_size "${BATCH_SIZE}" --lr 1e-5 --num_epochs 4
