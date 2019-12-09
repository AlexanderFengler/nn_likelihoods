# IMPORTS --------------------------------------------------------------------
# We are not importing tensorflow or keras here
# import tensorflow as tf
# from tensorflow import keras

import numpy as np
import yaml
import pandas as pd
from itertools import product
import multiprocessing as mp
import pickle
import uuid
import os
import sys
import argparse
import scipy as scp

# Sampler
from samplers import SliceSampler

# Analytical Likelihood for ddm
from cdwiener import batch_fptd

# Analytical Likelihood for lba
import clba

# Network converter
import keras_to_numpy as ktnp
# -----------------------------------------------------------------------------

# SUPPORT FUNCTIONS -----------------------------------------------------------
# Get full parameter vector including bounds
def make_parameter_bounds_for_sampler(mode = 'test',
                                      method_params = []):
    
    if mode == 'test':
        param_bounds = method_params['param_bounds_sampler'] + method_params['boundary_param_bounds_sampler']
    if mode == 'train':
        param_bounds = method_params['param_bounds_network'] + method_params['boundary_param_bounds_network']

    # If model is lba, lca, race we need to expand parameter boundaries to account for
    # parameters that depend on the number of choices
    if method == 'lba' or method == 'lca' or method == 'race':
        param_depends_on_n = method_params['param_depends_on_n_choice']
        param_bounds_tmp = []

        n_process_params = len(method_params['param_names'])

        p_cnt = 0
        for i in range(n_process_params):
            if method_params['param_depends_on_n_choice'][i]:
                for c in range(method_params['n_choices']):
                    param_bounds_tmp.append(param_bounds[i])
                    p_cnt += 1
            else:
                param_bounds_tmp.append(param_bounds[i])
                p_cnt += 1

        param_bounds_tmp += param_bounds[n_process_params:]
        return np.array(param_bounds_tmp)
    else: 
        return np.array(param_bounds)
# -----------------------------------------------------------------------------
    
# INITIALIZATIONS -------------------------------------------------------------
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--method",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--datatype",
                     type = str,
                     default = 'uniform') # real, uniform, perturbation experiment
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 1000)
    CLI.add_argument("--nmcmcsamples",
                     type = int,
                     default = 10000)
    CLI.add_argument("--outfileid",
                     type = str,
                     default = 'TEST')
    CLI.add_argument("--infileid",
                     type = str,
                     default = 'none')
    CLI.add_argument("--outfilesig",
                     type = str,
                     default = 'signature')
    CLI.add_argument("--boundmode",
                     type = str,
                     default = 'train')
    CLI.add_argument("--nchoices",
                     type = int,
                     default = 2)
    CLI.add_argument("--activedims",
                     nargs = "*",
                     type = int,
                     default = [0, 1, 2, 3])
    CLI.add_argument("--frozendims",
                     nargs = "*",
                     type = int,
                     default = [])
    CLI.add_argument("--frozendimsinit",
                     nargs = '*',
                     type = float,
                     default = [])
    
    args = CLI.parse_args()
    print(args)
    
    mode = args.boundmode
    machine = args.machine
    method = args.method
    analytic = ('analytic' in method)
    data_type = args.datatype
    n_samples = args.nsamples
    n_slice_samples = args.nmcmcsamples
    infile_id = args.infileid
    out_file_id = args.outfileid
    out_file_signature = args.outfilesig
    n_cpus = 'all'
    
    # Initialize the frozen dimensions
    if len(args.frozendims) > 1:
        frozen_dims = [[args.frozendims[i], args.frozendimsinit[i]] for i in range(len(args.frozendims))]
        active_dims = args.activedims
    else:
        active_dims = 'all'
        frozen_dims = 'none'
    
    if data_type == 'perturbation_experiment':
        file_ = 'base_data_perturbation_experiment_nexp_1_n_' + str(n_samples) + '_' + infile_id + '.pickle'
        out_file_signature = 'post_samp_perturbation_experiment_nexp_1_n_' + str(n_samples) + '_' + infile_id                                                                      
    if data_type == 'uniform':
        file_ = 'base_data_param_recov_unif_reps_1_n_' + str(n_samples) + '_' + infile_id + '.pickle'
        out_file_signature = 'post_samp_data_param_recov_unif_reps_1_n_' + str(n_samples) + '_' + infile_id
    
    if data_type == 'real':                                                                        
        file_ = args.infileid
        if machine == 'x7':
            data_folder = '/media/data_cifs/afengler/data/real/'
        if machine == 'ccv':
            data_folder = '/users/afengler/data/real/'
        
    if machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        output_folder = method_params['output_folder_x7']
        with open("model_paths_x7.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)[method]
            print(network_path)

    if machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        output_folder = method_params['output_folder']
        with open("model_paths.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)[method]
    
    method_params['n_choices'] = args.nchoices
    print(method_params)

    # Load weights, biases and activations of current network --------
    if analytic:
        pass
    else:
        with open(network_path + "weights.pickle", "rb") as tmp_file:
            weights = pickle.load(tmp_file)
            print(weights)
        with open(network_path + 'biases.pickle', 'rb') as tmp_file:
            biases = pickle.load(tmp_file)
            print(biases)
        with open(network_path + 'activations.pickle', 'rb') as tmp_file:
            activations = pickle.load(tmp_file)
            print(activations)
# ----------------------------------------------------------------

# DEFINE TARGET LIKELIHOODS FOR CORRESPONDING MODELS -------------------------------------------------
 
# ----------------------------------------------------------------------------------------------------

# MAKE PARAMETER / DATA GRID -------------------------------------------------------------------------
    # REFORMULATE param bounds
    if data_type == 'real':
        print(data_folder + file_)
        data = pickle.load(open(data_folder + file_, 'rb'))
        data_grid = data[0]
        param_grid = ['random' for i in range(data_grid.shape[0])]
    elif data_type == 'uniform':
        data = pickle.load(open(output_folder + file_, 'rb'))
        param_grid = data[0]
        data_grid = data[1]
    elif data_type == 'perturbation_experiment':
        pass # TODO fill in correct formatting for perturbation experiment
    else:
        print('Unknown Datatype, results will likely not make sense')   
        
    # Parameter bounds to pass to sampler    
    sampler_param_bounds = make_parameter_bounds_for_sampler(mode = mode, 
                                                             method_params = method_params)
    sampler_param_bounds = [sampler_param_bounds for i in range(data_grid.shape[0])]
    
    # 
    print('param_grid: ', param_grid)
    print('shape of data_grid:', data_grid.shape)
# ----------------------------------------------------------------------------------------------------

# RUN POSTERIOR SIMULATIONS --------------------------------------------------------------------------
   # MLP TARGET
    def mlp_target(params, data, 
                   ll_min= -16.11809 # corresponds to 1e-7
                  ): 
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
    def lba_target(params, data): # TODO add active and frozen dim vals
        return clba.batch_dlba2(rt = data[:, 0],
                                choice = data[:, 1],
                                v = params[:2],
                                A = params[2],
                                b = params[3], 
                                s = params[4],
                                ndt = params[5])

    # Define posterior samplers for respective likelihood functions
    def mlp_posterior(args): # args = (data, true_params)
        scp.random.seed()
        model = SliceSampler(bounds = args[2], 
                             target = mlp_target, 
                             w = .4 / 1024, 
                             p = 8,
                             data = args[0])
        
        model.sample(num_samples = n_slice_samples,
                     init = args[1],
                     active_dims = active_dims,
                     frozen_dim_vals = frozen_dims)
        return model.samples

    # Test navarro-fuss
    def nf_posterior(args): # TODO add active and frozen dim vals
        scp.random.seed()
        model = SliceSampler(bounds = args[2],
                             target = nf_target, 
                             w = .4 / 1024, 
                             p = 8,
                             active_dims = active_dims,
                             frozen_dim_vals = frozen_dims)
        
        model.sample(args[0], 
                     num_samples = n_slice_samples, 
                     init = args[1])
        return model.samples

    def lba_posterior(args):
        scp.random.seed()
        model = SliceSampler(bounds = args[2],
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
        posterior_samples = np.array(p.map(lba_posterior, zip(data_grid, param_grid, sampler_param_bounds)))
    elif method == 'ddm_analytic':
        posterior_samples = np.array(p.map(nf_posterior, zip(data_grid, param_grid, sampler_param_bounds)))
    else:
        posterior_samples = np.array(p.map(mlp_posterior, zip(data_grid, param_grid, sampler_param_bounds)))

    # Store files
    print('saving to file')
    print(output_folder + out_file_signature + '_' + out_file_id + ".pickle")
    pickle.dump((param_grid, data_grid, posterior_samples), 
                 open(output_folder + out_file_signature + '_' + out_file_id + ".pickle", "wb"))