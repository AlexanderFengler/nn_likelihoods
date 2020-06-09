#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J bb_dwnload

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/bb_download_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=30:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-24

# Run a command
#source /users/afengler/miniconda3/etc/profile.d/conda.sh
#conda activate tony

# source /users/afengler/.bashrc
# conda deactivate
# conda activate tf-cpu

# source /users/afengler/.bashrc
# conda deactivate
# conda activate tf-gpu-py37

# NNBATCH RUNS
python -u download_files_script.py --idx $SLURM_ARRAY_TASK_ID
