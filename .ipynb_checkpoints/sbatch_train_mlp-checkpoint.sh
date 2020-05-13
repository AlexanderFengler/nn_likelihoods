#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J mlp_train

# priority
##SBATCH --account=bibs-frankmj-condo
#SBATCH --account=carney-frankmj-condo

# email error reports
#SBATCH --mail-user=alexander_fengler@brown.edu 
#SBATCH --mail-type=ALL

# output file
#SBATCH --output /users/afengler/batch_job_out/mlp_train_ddm_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=24:00:00
#SBATCH --mem=320G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --constraint='quadrortx'
##SBATCH --constraint='cascade'
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-2

source /users/afengler/.bashrc
conda deactivate
conda activate tf-gpu-py37
# module load python/3.7.4 cuda/10.0.130 cudnn/7.4 tensorflow/2.0.0_gpu_py37

nfiles=150
method='ddm_analytic'
machine='ccv'
maxidfiles=300

#!/bin/bash
for i in {1..5}
do
   echo "Now starting run: $i \n"
   python -u /users/afengler/git_repos/nn_likelihoods/keras_fit_model.py --machine $machine --method $method --nfiles $nfiles --maxidfiles $maxidfiles --datafolder /users/afengler/data/kde/${method}/training_data_binned_0_nbins_0_n_20000/ --nbydataset 10000000 --warmstart 0
done

#!/bin/bash

# Pick number of files to consider
# nfiles=100
# method='levy'

# # Function call
# python -u keras_fit_model.py --machine x7 --method $method --nfiles $nfiles --datafolder /media/data_cifs/afengler/data/kde/${method}/training_data_binned_0_nbins_0_n_20000/ --nbydataset 10000000 --warmstart 0
#data_folder='/users/afengler/data/kde/levy/'


