#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J mc_weibull_ndt

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/weibull_mc_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH -c 12
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-100

# Run a command
#source /users/afengler/miniconda3/etc/profile.d/conda.sh
#conda activate tony
python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py ccv weibull_cdf_ndt uniform 2500 10000 $SLURM_ARRAY_TASK_ID
