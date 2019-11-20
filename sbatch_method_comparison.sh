#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J weibull_bg

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/weibull_bg_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH -c 12
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-50

# Run a command
#source /users/afengler/miniconda3/etc/profile.d/conda.sh
#conda activate tony
python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine ccv --method weibull_cdf --nmcmcsamples 50 --datatype real --infileid bg_stn_sampling_ready.pickle --boundmode train --outfilesig bg_stn_test --outfileid $SLURM_ARRAY_TASK_ID