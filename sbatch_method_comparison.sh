#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J angle_bg

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/angle_bg_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-100

# Run a command
#source /users/afengler/miniconda3/etc/profile.d/conda.sh
#conda activate tony

# python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine ccv --method angle --nmcmcsamples 1000 --datatype real --infileid bg_stn_sampling_ready.pickle --boundmode train --outfilesig bg_stn_posterior_samples --outfileid $SLURM_ARRAY_TASK_ID

# python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine ccv --method ddm --nmcmcsamples 100 --datatype perturbation_experiment --infileid 21 --boundmode train --outfilesig test_new_slice_sampler --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 3 --frozendims 2 --frozendimsinit 0.6 --samplerinit mle

# python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine ccv --method ornstein --nmcmcsamples 100 --datatype real --infileid bg_stn_sampling_ready.pickle --boundmode train --outfilesig _expanded_bounds_ --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 2 3 4 --samplerinit mle

# python -m cProfile -s tottime /media/data_cifs/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine x7 --method ornstein --nmcmcsamples 4000 --datatype real --infileid bg_stn_sampling_ready.pickle --boundmode train --outfilesig _expanded_bounds_test_ --outfileid 999 --activedims 0 1 2 3 4 --samplerinit random > profile_out.txt
#python -u /media/data_cifs/afengler/git_repos/nn_likelihoods/method_comparison_sim.py --machine x7 --method ornstein --nmcmcsamples 5000 --datatype real --infileid bg_stn_sampling_ready.pickle --boundmode train --outfilesig _expanded_bounds_test_ --outfileid 999 --activedims 0 1 2 3 4 --samplerinit random

nmcmcsamples=20000
nbyarrayjob=10
nsamples=2048

python -u method_comparison_sim.py --machine ccv --method ddm --nsamples $nsamples --nmcmcsamples $nmcmcsamples --datatype parameter_recovery --sampler diffevo --infileid 1 --boundmode train --outfilesig _expanded_bounds_test --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 2 3 4 --samplerinit mle --ncpus 10 --nbyarrayjob $nbyarrayjob #$SLURM_ARRAY_TASK_ID 

python -u method_comparison_sim.py --machine ccv --method full_ddm --nsamples $nsamples --nmcmcsamples $nmcmcsamples --datatype parameter_recovery --sampler diffevo --infileid 1 --boundmode train --outfilesig _expanded_bounds_test --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 2 3 4 5 6 --samplerinit mle --ncpus 10 --nbyarrayjob $nbyarrayjob #$SLURM_ARRAY_TASK_ID 

python -u method_comparison_sim.py --machine ccv --method angle --nsamples $nsamples --nmcmcsamples $nmcmcsamples --datatype parameter_recovery --sampler diffevo --infileid 1 --boundmode train --outfilesig _expanded_bounds_test --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 2 3 4 --samplerinit mle --ncpus 10 --nbyarrayjob $nbyarrayjob #$SLURM_ARRAY_TASK_ID 

python -u method_comparison_sim.py --machine ccv --method ornstein --nsamples $nsamples --nmcmcsamples $nmcmcsamples --datatype parameter_recovery --sampler diffevo --infileid 1 --boundmode train --outfilesig _expanded_bounds_test --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 2 3 4 --samplerinit mle --ncpus 10 --nbyarrayjob $nbyarrayjob #$SLURM_ARRAY_TASK_ID 

python -u method_comparison_sim.py --machine ccv --method weibull_cdf --nsamples $nsamples --nmcmcsamples $nmcmcsamples --datatype parameter_recovery --sampler diffevo --infileid 1 --boundmode train --outfilesig _expanded_bounds_test --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 2 3 4 5 --samplerinit mle --ncpus 10 --nbyarrayjob $nbyarrayjob #$SLURM_ARRAY_TASK_ID 

python -u method_comparison_sim.py --machine ccv --method levy --nsamples $nsamples --nmcmcsamples $nmcmcsamples --datatype parameter_recovery --sampler diffevo --infileid 1 --boundmode train --outfilesig _expanded_bounds_test --outfileid $SLURM_ARRAY_TASK_ID --activedims 0 1 2 3 4 --samplerinit mle --ncpus 10 --nbyarrayjob $nbyarrayjob #$SLURM_ARRAY_TASK_ID 

#--frozendims 2 --frozendimsinit 0.5