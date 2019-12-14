#!/bin/bash

# Get base simulations
jobID_1=$(sbatch  sbatch_base_simulations.sh | cut -f 4 -d' ')

# Get simulator stats
jobID_2=$(sbatch --dependency=afterok:$jobID_1 sbatch_sim_stats.sh | cut -f 4 -d' ')

# Make training data for filtered base simulations
sbatch  --dependency=afterany:$jobID_2 sbatch_kde_to_base_simulations.sh