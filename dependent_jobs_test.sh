#!/bin/bash

# first job - no dependencies
jobID_1=$(sbatch  sbatch_method_comparison.sh | cut -f 4 -d' ')

# second job - depends on job1
jobID_2=$(sbatch --dependency=afterok:$jobID_1 sbatch_method_comparison.sh | cut -f 4 -d' ')

# # third job - depends on job2
# sbatch  --dependency=afterany:$jobID_2  job3.sh