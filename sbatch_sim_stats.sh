#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J tpl_2_weibull

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/angle_ndt_sim_stats_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=1-100

# Run a command
python -u simulator_get_stats.py --machine ccv --method angle --simfolder training_data_binned_0_nbins_0_n_20000 --fileprefix angle_nchoices_2_train_data_binned_0_nbins_0_n_20000 --fileid $SLURM_ARRAY_TASK_ID

#$SLURM_ARRAY_TASK_ID

