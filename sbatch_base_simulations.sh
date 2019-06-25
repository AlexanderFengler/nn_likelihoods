#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
##SBATCH -J ornstein_base_sim
##SBATCH -J ddm_base_sim
##SBATCH -J full_ddm_base_sim
##SBATCH -J weibull_base_sim

# output file
##SBATCH --output /users/afengler/batch_job_out/full_ddm_base_sim_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/ddm_base_sim_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/ornstein_base_sim_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/weibull_base_sim_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=15:00:00
#SBATCH --mem=4G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=1-50

# Run a command

python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py
