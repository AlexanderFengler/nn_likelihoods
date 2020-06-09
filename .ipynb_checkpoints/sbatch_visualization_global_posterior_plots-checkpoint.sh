#!/bin/bash


# DDM -----------------------------------------------
model="ddm"
machine="home"
method="mlp"
traindattype="kde"
networkidx=8
n=( 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done


model="ddm"
machine="home"
method="mlp"
traindattype="analytic"
networkidx=2
n=( 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done


model="ddm"
machine="home"
method="mlp"
traindattype="analytic"
networkidx=2
n=( 4096 )
analytic=1
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done
# ---------------------------------------------------------


# DDM_SDV -------------------------------------------------
model="ddm_sdv"
machine="home"
method="mlp"
traindattype="kde"
networkidx=2
n=( 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done


model="ddm_sdv"
machine="home"
method="mlp"
traindattype="analytic"
networkidx=2
n=( 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done


model="ddm_sdv"
machine="home"
method="mlp"
traindattype="analytic"
networkidx=2
n=( 4096 )
analytic=1
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done
# ----------------------------------------------------------

# ANGLE2 ---------------------------------------------------
model="angle2"
machine="home"
method="mlp"
traindattype="kde"
networkidx=-1
n=( 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done
# -------------------------------------------------------------

# FULL_DDM2 ---------------------------------------------------
model="full_ddm2"
machine="home"
method="mlp"
traindattype="kde"
networkidx=-1
n=( 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done
# -----------------------------------------------------------

# ORNSTEIN ---------------------------------------------------
model="ornstein"
machine="home"
method="mlp"
traindattype="kde"
networkidx=-1
n=( 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done
# -----------------------------------------------------------

# WEIBULL CDF ---------------------------------------------------
model="weibull_cdf"
machine="home"
method="mlp"
traindattype="kde"
networkidx=-1
n=( 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair
done

# -----------------------------------------------------------