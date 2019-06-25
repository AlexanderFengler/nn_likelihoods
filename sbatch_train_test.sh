#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
##SBATCH -J ddm_train_test
##SBATCH -J full_ddm_train_test
##SBATCH -J weibull_train_test
##SBATCH -J ornstein_train_test

# output file
##SBATCH --output /users/afengler/batch_job_out/ddm_train_test_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/full_ddm_train_test_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/weibull_train_test_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/ornstein_train_test_%A_%a.out


# Request runtime, memory, cores:
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=1-25

# Run a command

python -u /users/afengler/git_repos/nn_likelihoods/kde_train_test.py
