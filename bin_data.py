# Environ
import scipy as scp
#import tensorflow as tf
#from scipy.stats import gamma
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.neighbors import KernelDensity
#import random
#import multiprocessing as mp
#import psutil
import pickle
import os
#import re

# Own
#import ddm_data_simulation as ds
#import cddm_data_simulation as cds
#import kde_training_utilities as kde_util
#import kde_class as kde
#import boundary_functions as bf


def bin_simulator_output(out = [0, 0],
                         bin_dt = 0.04,
                         eps_correction = 1e-7,
                         params = ['v', 'a', 'w', 'ndt']):

    # hardcode 'max_t' to 20sec for now
    n_bins = int(20.0 / bin_dt + 1)
    #n_bins = int(out[2]['max_t'] / bin_dt + 1)
    bins = np.linspace(0, out[2]['max_t'], n_bins)
    counts = []
    counts.append(np.histogram(out[0][out[1] == 1], bins = bins)[0] / out[2]['n_samples'])
    counts.append(np.histogram(out[0][out[1] == -1], bins = bins)[0] / out[2]['n_samples'])

    n_small = 0
    n_big = 0

    for i in range(len(counts)):
        n_small += sum(counts[i] < eps_correction)
        n_big += sum(counts[i] >= eps_correction)

    for i in range(len(counts)):
        counts[i][counts[i] <= eps_correction] = eps_correction
        counts[i][counts[i] > eps_correction] = counts[i][counts[i] > eps_correction] - (eps_correction * (n_small / n_big))    

    for i in range(len(counts)):
        counts[i] =  np.asmatrix(counts[i]).T

    label = np.concatenate(counts, axis = 1)
    features = [out[2]['v'], out[2]['a'], out[2]['w']], out[2]['ndt']]
    return (features, label)

files_ = os.listdir('/users/afengler/data/kde/ddm/base_simulations_ndt_20000')
labels = np.zeros((len(files) - 2, 500, 2))
features = np.zeros((len(files) - 2, 3))
   
cnt = 0
i = 0
file_dim = 100
for file_ in files_[:1000]:
    if file_[:8] == 'ddm_flex':
        out = pickle.load(open('/users/afengler/data/kde/ddm/base_simulations_ndt_20000' + file_, 'rb'))
        features[cnt], labels[cnt] = bin_simulator_output(out = out)
        if cnt % file_dim == 0:
            print(cnt)
            pickle.dump((labels[(i * file_dim):((i + 1) * file_dim)], features[(i * file_dim):((i + 1) * file_dim)]), open('/users/afengler/data/kde/ddm/base_simulations_ndt_20000_binned/dataset_' + str(i), 'wb'))
            i += 1
        cnt += 1




