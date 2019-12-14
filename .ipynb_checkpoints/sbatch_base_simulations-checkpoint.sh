#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J ornstein_sim

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/ornstein_sim_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=1-100

# --------------------------------------------------------------------------------------
# Sequentially run different kind of models

# declare -a dgps=( "ddm" "full_ddm" "angle" "weibull_cdf" "ornstein" "lca" "race_model" ) 
# n_samples=( 128 256 512 1024 2048 4096 8192 ) #( 50000 100000 200000 400000 )
# n_choices=( 2 3 4 ) #( 4 5 6 )
# n_parameter_sets=100  #20000
# n_bins=( 256 512 )
# # outer -------------------------------------
# for bins in "${n_bins[@]}"
# do
#     for n in "${n_samples[@]}"
#     do
#     # inner -------------------------------------
#         for dgp in "${dgps[@]}"
#         do
#             if [[ "$dgp" = "lca" ]] || [[ "$dgp" = "race_model" ]];
#             then
#                 for n_c in "${n_choices[@]}"
#                     do
#                        python -u dataset_generator.py --machine x7 --dgplist $dgp --datatype parameter_recovery --nreps 10 --binned 1 --nbins $bins --maxt 10 --nchoices $n_c --nsamples $n --mode cnn --nparamsets $n_parameter_sets --save 1 --deltat 0.001 #--fileid $SLURM_ARRAY_TASK_ID 
#                        echo "$dgp"
#                        echo $n_c
#                 done
#             else
#                  python -u dataset_generator.py --machine x7 --dgplist $dgp --datatype parameter_recovery --nreps 10 --binned 1 --nbins $bins --maxt 10 --nchoices 2 --nsamples $n --mode cnn --nparamsets $n_parameter_sets --save 1  --deltat 0.001 #--fileid $SLURM_ARRAY_TASK_ID
#                  echo "$dgp"
#                  echo $n_c
#             fi
#         done
#                 # normal call to function
#     done
# done
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# Run rdgp
# n_parameter_sets=20000
# bins=256
# n_c=2

# python -u dataset_generator.py --machine ccv --dgplist angle --datatype r_dgp --nreps 1 --binned 1 
# --nbins $bins --maxt 10 --nchoices $n_c --nsimbnds 100 100000 --mode cnn --nparamsets $n_parameter_sets --save 1 --deltat 0.001 --fileid $SLURM_ARRAY_TASK_ID 

# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# Base data for mlp
n_parameter_sets=100
bins=256
n_c=2

python -u dataset_generator.py --machine ccv --dgplist weibull_cdf --datatype cnn_train --nreps 1 --binned 0 --nbins 0 --maxt 20 --nchoices $n_c --nsamples 200 --mode mlp --nparamsets $n_parameter_sets --save 1 --deltat 0.001 --fileid $SLURM_ARRAY_TASK_ID 

#---------------------------------------------------------------------------------------

# python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv ornstein 2 100000 1 $SLURM_ARRAY_TASK_ID base_simulations 10000 0

# python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv lca 3 100000 1 $SLURM_ARRAY_TASK_ID base_simulations 10000 0

# python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv lca 4 100000 1 $SLURM_ARRAY_TASK_ID base_simulations 10000 0

# python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv lca 5 100000 1 $SLURM_ARRAY_TASK_ID base_simulations 10000 0

# python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv lca 6 100000 1 $SLURM_ARRAY_TASK_ID base_simulations 10000 0

#python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 3 100000 1 $SLURM_ARRAY_TASK_ID 1000 0

#python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 5 100000 1 $SLURM_ARRAY_TASK_ID

#python -u /users/afengler/git_repos/nn_likelihoods/kde_base_simulations.py ccv race_model 6 100000 1 $SLURM_ARRAY_TASK_ID