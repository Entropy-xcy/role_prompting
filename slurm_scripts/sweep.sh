#!/bin/bash

#SBATCH --job-name=role-prompt-sweep
#SBATCH --output=slurm_outputs/output-%j.txt
#SBATCH --error=slurm_outputs/error-%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A5000:1
#SBATCH --partition=compsci-gpu

# Array of different argument values
OCCUPATIONS=("lawyer" "doctor" "professor" "student")
EDUCATIONS=("PhD" "Masters" "Bachelors" "High-School" "Primary-School")
GENDERS=("Female" "Male" "Non-binary")
AGES=("child" "teenager" "adult" "elderly")
NATIONALITIES=("American" "Chinese")

for occupation in "${OCCUPATIONS[@]}"; do
    for education in "${EDUCATIONS[@]}"; do
        for gender in "${GENDERS[@]}"; do
            for age in "${AGES[@]}"; do
                for nationality in "${NATIONALITIES[@]}"; do
                    # Submit the job
                    sbatch slurm_scripts/eval_one.sh "$occupation" "$education" "$gender" "$age" "$nationality"
                done
            done
        done
    done
done
