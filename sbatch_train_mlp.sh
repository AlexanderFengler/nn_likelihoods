#!/bin/bash

# Pick number of files to consider
nfiles=100

# Function call
python -u keras_fit_model.py --machine x7 --method angle --nfiles $nfiles --datafolder /media/data_cifs/afengler/data/kde/angle/training_data_binned_0_nbins_0_n_20000/ --nbydataset 10000000 --warmstart 0 