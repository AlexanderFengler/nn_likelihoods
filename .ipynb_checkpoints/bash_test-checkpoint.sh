#!/bin/bash

method='weibull_cdf_concave'
trainn=100000
base_folder="users/afengler/data/analytic/"
data_folder="$base_folder${method}/training_data_binned_0_nbins_0_n_${trainn}/"

#base_data_folder += ${method}
echo "$data_folder"