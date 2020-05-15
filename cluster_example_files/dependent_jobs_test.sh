#!/bin/bash

# Get base simulations
jobID_1=$(sbatch sbatch_base_simulations.sh | cut -f 4 -d' ')

# Get simulator stats
jobID_2=$(sbatch --dependency=afterok:$jobID_1 sbatch_sim_stats.sh | cut -f 4 -d' ')

# Make training data for filtered base simulations
jobID_3=$(sbatch --dependency=afterok:$jobID_2 sbatch_kde_to_base_simulations.sh | cut -f 4 -d' ')


# Run complete training and sampling pipeline
jobID_1=$(sbatch /users/afengler/git_repos/nn_likelihoods/sbatch_train_mlp.sh | cut -f 4 -d ' ')
jobID_2=$(sbatch --dependency=afterok:$jobID_1 /users/afenger/git_repos/nn_likelihoods/sbatch_train_mlp.sh  | cut -f 4 -d ' ')
jobID_3=$(sbatch --dependency=afterok:$jobID_2 /users/afengler/git_repos/nn_likelihoods/sbatch_parameter_recovery.sh | cur -f 4 -d ' ')

sbatch --dependency=afterok:$jobID_3 /users/afengler/git_repos/nn_likelihoods/sbatch_method_comparison.sh 