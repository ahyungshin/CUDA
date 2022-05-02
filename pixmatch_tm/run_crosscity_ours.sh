#!/bin/bash

#SBATCH --job-name=target2_optimizer_no_loss
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --time=14-0


source /data/seunan/init.sh
conda activate torch38gpu

HYDRA_FULL_ERROR=1 python main.py --config-name=gta5 lam_aug=0.10 name=gta52rio
