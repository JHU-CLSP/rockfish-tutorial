#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=12:00:00
#SBATCH --qos=qos_gpu
#SBATCH --job-name="test_gpus"
#SBATCH --output="result.txt" # Path to store logs


module load anaconda
module load cuda/11.6.0

### init virtual environment if needed
# conda create -n toy_classification_env python=3.7


### see the other environments
# conda info --envs

conda activate toy_classification_env
pip install torch
srun python example.py

