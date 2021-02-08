#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J mlp_analytic

# priority
##SBATCH --account=bibs-frankmj-condo
#SBATCH --account=carney-frankmj-condo

# email error reports
#SBATCH --mail-user=alexander_fengler@brown.edu 
#SBATCH --mail-type=ALL

# output file
#SBATCH --output /users/afengler/batch_job_out/mlp_train_ddm_analytic_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=18:00:00
#SBATCH --mem=192G
#SBATCH -c 8
#SBATCH -N 1
##SBATCH --constraint='quadrortx'
##SBATCH --constraint='cascade'
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-1

source /users/afengler/.bashrc
conda deactivate
conda activate tf-gpu-py37
module load cuda/10.0.130
module load cudnn/7.6

nfiles=200
method='weibull_cdf_concave'
analytic=0 # Training labels from analytic likelihood (1) or from KDE (0) (This is now all in the model_name)
machine='ccv'
maxidfiles=200
trainn=100000

#base_folder="users/afengler/data/analytic/" # Base folder where data sits (finally subfolder targeted) CHANGE THIS
# training_data_folder="$base_folder/${method}/training_data_binned_0_nbins_0_n_${trainn}/" # subfolder with self explanatory title SUBFOLDER STRUCTURE SHOULD BE MAINTAINED

training_data_folder="training_data_binned_0_nbins_0_n_${trainn}/"
#model_data_folder="$base_folder/${method}/" # MAINTAIN

if [ $analytic -eq 1 ]; then
    for i in {1..2}
    do
       echo "Now starting run: $i"
       python -u /users/afengler/git_repos/nn_likelihoods/keras_fit_model.py --machine $machine \
                                                                             --method $method \
                                                                             --nfiles $nfiles \
                                                                             --maxidfiles $maxidfiles \
                                                                             --traindatafolder $training_data_folder \
                                                                             --warmstart 0 \
                                                                             --analytic $analytic
    done
else
    for i in {1..2}
    do
       echo "Now starting run: $i"
       python -u /users/afengler/git_repos/nn_likelihoods/keras_fit_model.py --machine $machine \
                                                                             --method $method \
                                                                             --nfiles $nfiles \
                                                                             --traindatafolder $training_data_folder \
                                                                             --maxidfiles $maxidfiles \
                                                                             --warmstart 0 \
                                                                             --analytic $analytic
    done

fi

# USE THIS DATAFOLDER FOR ANY CASE UP TO DDM ANALYTIC
# datafolder=/users/afengler/data/kde/${method}/training_data_binned_0_nbins_0_n_20000/

# for i in {1..5}
# do
#    echo "Now starting run: $i \n"
#    python -u /users/afengler/git_repos/nn_likelihoods/keras_fit_model.py --machine $machine --method $method --nfiles $nfiles --maxidfiles $maxidfiles --datafolder /users/afengler/data/${model_type}/${method}/training_data_binned_0_nbins_0_n_20000/ --nbydataset 10000000 --warmstart 0
# done

# #!/bin/bash


# Pick number of files to consider
# nfiles=100
# method='levy'

# # Function call
# python -u keras_fit_model.py --machine x7 --method $method --nfiles $nfiles --datafolder /media/data_cifs/afengler/data/kde/${method}/training_data_binned_0_nbins_0_n_20000/ --nbydataset 10000000 --warmstart 0
#data_folder='/users/afengler/data/kde/levy/'