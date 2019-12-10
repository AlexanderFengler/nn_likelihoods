#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J angle_bg

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/angle_bg_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH -c 12
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-25

# Run a command
#source /users/afengler/miniconda3/etc/profile.d/conda.sh
#conda activate tony

# python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine ccv --method angle --nmcmcsamples 1000 --datatype real --infileid bg_stn_sampling_ready.pickle --boundmode train --outfilesig bg_stn_posterior_samples --outfileid $SLURM_ARRAY_TASK_ID

# python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine ccv --method ddm --nmcmcsamples 100 --datatype perturbation_experiment --infileid 21 --boundmode train --outfilesig test_new_slice_sampler --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 3 --frozendims 2 --frozendimsinit 0.6 --samplerinit mle

python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine ccv --method weibull_cdf --nmcmcsamples 10000 --datatype real --infileid bg_stn_sampling_ready.pickle --boundmode train --outfilesig _fix_w_ --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 3 4 5 --frozendims 2 --frozendimsinit 0.5 --samplerinit mle