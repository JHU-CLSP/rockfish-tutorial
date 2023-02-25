#!/bin/bash
#SBATCH -A danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=12:00:00
#SBATCH --output="/home/danielk/classification_example/out" # Path to store logs
module load anaconda
module load cuda/11.6.0

## You can see the available modules with:
# module avail

### init virtual environment if needed
# conda create -n toy_classification_env python=3.7
#conda info --envs
#conda activate toy_classification_env
pip install -r requirements.txt
#srun python classification.py
srun python check_gpu.py


