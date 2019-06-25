#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
##SBATCH -J ornstein_sim_stats
##SBATCH -J ddm_sim_stats
##SBATCH -J full_ddm_sim_stats
##SBATCH -J weibull_sim_stats

# output file
##SBATCH --output /users/afengler/batch_job_out/ddm_sim_stats_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/full_ddm_sim_stats_%A_%a.out
#SBATCH --output /users/afengler/batch_job_out/ornstein_sim_stats_%A_%a.out
##SBATCH --output /users/afengler/batch_job_out/weibull_sim_stats_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=15:00:00
#SBATCH --mem=16G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=1-1

# Run a command
python -u /users/afengler/git_repos/nn_likelihoods/simulator_get_stats.py
