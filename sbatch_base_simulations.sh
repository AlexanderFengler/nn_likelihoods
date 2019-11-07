#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J full_ddm_sim

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/full_ddm_sim_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=1-100

# Run a command

# n_data_points=( 100 200 400 800 1600 3200 6400 12800 25600 )
# # outer -------------------------------------
# for n in "${n_data_points[@]}"
# # inner -------------------------------------
# do
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 3 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 4 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 5 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 6 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv ddm 2 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv angle 2 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 2 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv weibull_cdf 2 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv ornstein 2 $n 1 $SLURM_ARRAY_TASK_ID 1000 0
#     python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv full_ddm 2 $n 1 $SLURM_ARRAY_TASK_ID 1000 0




# #         echo $n
# #         echo $id
# done
# # -------------------------------------------
# #done
# # -------------------------------------------

python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv ornstein 2 100000 1 $SLURM_ARRAY_TASK_ID base_simulations 10000 0

#python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 3 100000 1 $SLURM_ARRAY_TASK_ID 1000 0

#python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 5 100000 1 $SLURM_ARRAY_TASK_ID

#python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 6 100000 1 $SLURM_ARRAY_TASK_ID