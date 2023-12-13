#!/bin/bash

#SBATCH --job-name=role-prompt-sweep
#SBATCH --output=slurm_outputs/output-%j.txt
#SBATCH --error=slurm_outputs/error-%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A5000:1
#SBATCH --partition=compsci-gpu

# Run the script with passed parameters
python lm_eval_role_baseline.py 
