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

srun python classification.py --device cuda --model "distilbert-base-uncased" --batch_size "256" --lr 1e-5 --num_epochs 20
# srun python classification.py --device cuda --model "bert-base-uncased" --batch_size "128" --lr 1e-5 --num_epochs 20
# srun python classification.py --device cuda --model "bert-large-uncased" --batch_size "64" --lr 1e-5 --num_epochs 4
# srun python classification.py --device cuda --model "roberta-base" --batch_size "128" --lr 1e-4 --num_epochs 20
# srun python classification.py --device cuda --model "roberta-large" --batch_size "64" --lr 1e-5 --num_epochs 4
# srun python classification.py --device cuda --model "t5-small" --batch_size "256" --lr 1e-5 --num_epochs 4
# srun python classification.py --device cuda --model "t5-base" --batch_size "256" --lr 1e-5 --num_epochs 4
# srun python classification.py --device cuda --model "t5-large" --batch_size "64" --lr 1e-5 --num_epochs 4
# srun python classification.py --device cuda --model "t5-3b" --batch_size "16" --lr 1e-5 --num_epochs 10
# srun python classification.py --device cuda --model "t5-11b" --batch_size "4" --lr 1e-5 --num_epochs 4