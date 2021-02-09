#!/bin/bash
declare -a dgps=( $@ )
# first job - no dependencies
jobID_1=$(sbatch  sbatch_base_simulations_full.sh ${dgps[@]}| cut -f 4 -d' ')

# second job - depends on job1
for dgp in "${dgps[@]}"
do
    sbatch --dependency=afterok:$jobID_1 sbatch_train_mlp_cpu.sh $dgp
done
# third job - depends on job2
# sbatch  --dependency=afterany:$jobID_2  job3.sh