#!/bin/bash


plotlist=( "hdi_p" "hdi_coverage" "parameter_recovery_scatter") #( "posterior_variance" "hdi_coverage" "hdi_p" "parameter_recovery_scatter" "parameter_recovery_hist" "posterior_pair" "model_uncertainty" "posterior_predictive" )

# DDM -----------------------------------------------
model="ddm"
machine="home"
method="mlp"
traindattype="kde"
networkidx=8
n=( 1024 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done


model="ddm"
machine="home"
method="mlp"
traindattype="analytic"
networkidx=2
n=( 1024 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done


model="ddm"
machine="home"
method="navarro"
traindattype="analytic"
networkidx=2
n=( 1024 4096 )
analytic=1
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done
# ---------------------------------------------------------


# DDM_SDV -------------------------------------------------

model="ddm_sdv"
machine="home"
method="mlp"
traindattype="kde"
networkidx=2
n=( 1024 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done


model="ddm_sdv"
machine="home"
method="mlp"
traindattype="analytic"
networkidx=2
n=( 1024 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done


model="ddm_sdv"
machine="home"
method="navarro"
traindattype="analytic"
networkidx=2
n=( 1024 4096 )
analytic=1
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done
----------------------------------------------------------

# ANGLE2 ---------------------------------------------------
model="angle2"
machine="home"
method="mlp"
traindattype="kde"
networkidx=-1
n=( 1024 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done
# # -------------------------------------------------------------

# # FULL_DDM2 ---------------------------------------------------
# model="full_ddm2"
# machine="home"
# method="mlp"
# traindattype="kde"
# networkidx=-1
# n=( 4096 )
# analytic=0
# rhatcutoff=1.1
# npostpred=9
# npostpair=9

# for n_tmp in "${n[@]}"
# do
#     python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
# done
# # -----------------------------------------------------------

# # ORNSTEIN ---------------------------------------------------
model="ornstein"
machine="home"
method="mlp"
traindattype="kde"
networkidx=-1
n=( 1024 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done
# # -----------------------------------------------------------

# # WEIBULL CDF ---------------------------------------------------
model="weibull_cdf2"
machine="home"
method="mlp"
traindattype="kde"
networkidx=-1
n=( 1024 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done

# # -----------------------------------------------------------

# LEVY ------------------------------------------------------------
model="levy"
machine="home"
method="mlp"
traindattype="kde"
networkidx=-1
n=( 1024 4096 )
analytic=0
rhatcutoff=1.1
npostpred=9
npostpair=9
# plotlist=( "posterior_variance" "hdi_coverage" "hdi_p" "parameter_recovery_scatter" "parameter_recovery_hist" "posterior_pair" "model_uncertainty" "posterior_predictive" )

for n_tmp in "${n[@]}"
do
    python -u visualization_global_posterior_plots.py --model $model --machine $machine --method $method --networkidx $networkidx --traindattype $traindattype --n $n_tmp --analytic $analytic --rhatcutoff $rhatcutoff --npostpred $npostpred --npostpair $npostpair --plots ${plotlist[@]}
done
# # -----------------------------------------------------------

