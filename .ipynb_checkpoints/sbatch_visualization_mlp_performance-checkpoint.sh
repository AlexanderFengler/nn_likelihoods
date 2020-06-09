#!/bin/bash

model="weibull_cdf"
machine="home"
traindatanalytic=0
ngraphs=9
trainfileidx=0
networkidx=-1
mlekdereps=1
manifoldlayers=50

python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers