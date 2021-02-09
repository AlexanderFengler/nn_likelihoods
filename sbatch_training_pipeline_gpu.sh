#!/bin/bash

# The data generating processes are passed as arguments separated by a space
declare -a dgps=( $@ )

# Running the simulators
jobID_1=$(sbatch  sbatch_data_generator.sh ${dgps[@]}| cut -f 4 -d' ')

# Running the network training
for dgp in "${dgps[@]}"
do
    sbatch --dependency=afterok:$jobID_1 sbatch_train_mlp_gpu.sh $dgp
done

# third job - depends on job2
# sbatch  --dependency=afterany:$jobID_2  job3.sh