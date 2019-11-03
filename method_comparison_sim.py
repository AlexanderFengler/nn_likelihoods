# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import yaml
import pandas as pd
from itertools import product
from samplers import SliceSampler
import pickle
import uuid
import os
import sys

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from cdwiener import batch_fptd
import clba

import keras_to_numpy as ktnp

# INITIALIZATIONS -------------------------------------------------------------
machine = sys.argv[1] # 'ccv', 'x7'
method = sys.argv[2]
analytic = ('analytic' in method)
data_type = sys.argv[3]
n_data_samples = sys.argv[4]
n_slice_samples = sys.argv[5]
file_id = sys.argv[6]
n_cpus = 'all'

#n_sims = 10
#n_data_samples = int(sys.argv[2])

print('argument list: ', str(sys.argv))
#out_file_signature 

if data_type == 'perturbation_experiment':
    file_ = 'base_data_perturbation_experiment_nexp_1_n_' + n_data_samples + '_' + file_id + '.pickle'
    out_file_signature = 'post_samp_perturbation_experiment_nexp_1_n_' + n_data_samples + '_' + file_id
if data_type == 'uniform':
    file_ = 'base_data_param_recov_unif_reps_1_n_' + n_data_samples + '_' + file_id + '.pickle'
    out_file_signature = 'post_samp_data_param_recov_unif_reps_1_n_' + n_data_samples + '_' + file_id

if machine == 'x7':
    method_comparison_folder = "/media/data_cifs/afengler/data/kde/ddm/method_comparison/"
if machine == 'ccv':
    method_comparison_folder = '/users/afengler/data/kde/ddm/method_comparison/'

if machine == 'x7':
    stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    output_folder = method_params['output_folder_x7']
    with open("model_paths_x7.yaml") as tmp_file:
        network_path = yaml.load(tmp_file)[method]

if machine == 'ccv':
    stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    output_folder = method_params['output_folder']
    with open("model_paths.yaml") as tmp_file:
        network_path = yaml.load(tmp_file)[method]
        
print(stats)
print(method_params)

# Load weights, biases and activations of current network --------
if analytic:
    pass
else:
    with open(network_path + "weights.pickle", "rb") as tmp_file:
        weights = pickle.load(tmp_file)
    with open(network_path + 'biases.pickle', 'rb') as tmp_file:
        biases = pickle.load(tmp_file)
    with open(network_path + 'activations.pickle', 'rb') as tmp_file:
        activations = pickle.load(tmp_file)
# ----------------------------------------------------------------

# DEFINE TARGET LIKELIHOODS FOR CORRESPONDING MODELS -------------------------------------------------

# MLP TARGET
def mlp_target(params, data, likelihood_min = 1e-7): 
    ll_min = np.log(likelihood_min)
    params_rep = np.tile(params, (data.shape[0], 1))
    input_batch = np.concatenate([params_rep, data], axis = 1)
    out = np.maximum(ktnp.predict(input_batch, weights, biases, activations), ll_min)
    return np.sum(out)

# NAVARRO FUSS (DDM)
def nf_target(params, data, likelihood_min = 1e-48):
    return np.sum(np.maximum(np.log(batch_fptd(data[:, 0] * data[:, 1] * (- 1),
                                               params[0],
                                               params[1] * 2, 
                                               params[2],
                                               params[3])), np.log(likelihood_min)))

# LBA ANALYTIC 
def lba_target(params, data):
    return clba.batch_dlba2(rt = data[:, 0], 
                            choice = data[:, 1], 
                            v = params[:2],
                            A = params[2], 
                            b = params[3], 
                            s = params[4],
                            ndt = params[5])

# ----------------------------------------------------------------------------------------------------

# MAKE PARAMETER / DATA GRID -------------------------------------------------------------------------
# REFORMULATE param bounds
data = pickle.load(open(method_comparison_folder + file_, 'rb'))
        
param_grid = data[0]
data_grid = data[1]

print('param_grid: ', param_grid)
print('shape of data_grid:', data_grid.shape)
# ----------------------------------------------------------------------------------------------------

# RUN POSTERIOR SIMULATIONS --------------------------------------------------------------------------

# Get full parameter vector including bounds
if method[:3] == 'lba':
    sampler_param_bounds = np.array(method_params["param_bounds_sampler"] + method_params["boundary_param_bounds_sampler"])
else:
    sampler_param_bounds = np.array(method_params["param_bounds_sampler"] + method_params["boundary_param_bounds_sampler"])

# Define posterior samplers for respective likelihood functions
def mlp_posterior(args): # args = (data, true_params)
    model = SliceSampler(bounds = sampler_param_bounds, 
                         target = mlp_target, 
                         w = .4 / 1024, 
                         p = 8)
    model.sample(args[0], num_samples = n_slice_samples, init = args[1])
    return model.samples

#test navarro-fuss
def nf_posterior(args):
    model = SliceSampler(bounds = sampler_param_bounds,
                         target = nf_target, 
                         w = .4 / 1024, 
                         p = 8)
    model.sample(args[0], num_samples = n_slice_samples, init = args[1])
    return model.samples

def lba_posterior(args):
    model = SliceSampler(bounds = sampler_param_bounds,
                         target = lba_target,
                         w = .4 / 1024,
                         p = 8)
    model.sample(args[0], num_samples = n_slice_samples, init = args[1])
    return model.samples

# Make available the specified amount of cpus
if n_cpus == 'all':
    p = mp.Pool(mp.cpu_count())
    
else: 
    p = mp.Pool(n_cpus)

# Run the sampler with correct target as specified above
if method == 'lba_analytic':
    posterior_samples = np.array(p.map(lba_posterior, zip(data_grid, param_grid)))
elif method == 'ddm_analytic':
    posterior_samples = np.array(p.map(nf_posterior, zip(data_grid, param_grid)))
else:
    posterior_samples = np.array(p.map(mlp_posterior, zip(data_grid, param_grid)))

# Store files
pickle.dump((param_grid, data_grid, posterior_samples), 
            open(output_folder + out_file_signature + ".pickle", "wb"))
