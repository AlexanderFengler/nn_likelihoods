#!/bin/bash
declare -a dgps=( $@ )
method='weibull_cdf_concave'
trainn=100000
base_folder="users/afengler/data/analytic/"
data_folder="$base_folder${method}/training_data_binned_0_nbins_0_n_${trainn}/"

bash bash_test_level_2.sh ${dgps[@]}
#base_data_folder += ${method}
echo "$data_folder"
echo "$dgps"
maxnum=10

for ((i = 1; i <= $maxnum; i++))
    do 
        echo "NOW TRAINING NETWORK: $i of 10"
        echo ${@:1:2}
    done

for j in "${dgps[@]}"
do
    echo "$j"
done