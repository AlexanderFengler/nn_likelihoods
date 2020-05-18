# IMPORTS --------------------------------------------------------------------
# We are not importing tensorflow or keras here
# import tensorflow as tf
# from tensorflow import keras
import os
import time
import re
#os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

from numpy import ndarray
import numpy as np
import yaml
import pandas as pd
from itertools import product
import multiprocessing as mp
import pickle
import uuid

import sys
import argparse
import scipy as scp
from scipy.optimize import differential_evolution

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model
# os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Sampler
from samplers import SliceSampler
from samplers import DifferentialEvolutionSequential
# Analytical Likelihood for ddm
from cdwiener import batch_fptd

# Analytical Likelihood for lba
import clba

# Network converter
#import keras_to_numpy as ktnp
import ckeras_to_numpy as ktnp

import keras_to_numpy_class as ktnpc


# Tensorflow 
import tensorflow as tf
from tensorflow import keras
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
                     default = 'parameter_recovery') # real, parameter_recovery, perturbation experiment
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 1000)
    CLI.add_argument("--nmcmcsamples",
                     type = int,
                     default = 10000)
    CLI.add_argument("--sampler",
                    type = str,
                    default = 'slice')
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
    CLI.add_argument("--samplerinit",
                     type = str,
                     default = 'mle') # 'mle', 'random', 'true'
    CLI.add_argument("--nbyarrayjob",
                     type = int,
                     default = 10)
    CLI.add_argument("--ncpus",
                     type = int,
                     default = 10)
    CLI.add_argument("--nnbatchid",  # nnbatchid is used if we use the '_batch' parts of the model_path files (essentially to for pposterior sample runs that check if for the same model across networks we observe similar behavior)
                     type = int,
                     default = -1)
    CLI.add_argument("--analytic",
                     type = int,
                     default = 0)
    
    args = CLI.parse_args()
    print(args)
    
    mode = args.boundmode
    machine = args.machine
    method = args.method
    analytic = ('analytic' in method)
    sampler = args.sampler
    data_type = args.datatype
    n_samples = args.nsamples
    n_slice_samples = args.nmcmcsamples
    infile_id = args.infileid
    out_file_id = args.outfileid
    out_file_signature = args.outfilesig
    n_cpus = args.ncpus
    n_by_arrayjob = args.nbyarrayjob
    nnbatchid = args.nnbatchid
    analytic= args.analytic

    # Initialize the frozen dimensions
    if len(args.frozendims) >= 1:
        frozen_dims = [[args.frozendims[i], args.frozendimsinit[i]] for i in range(len(args.frozendims))]
        active_dims = args.activedims
    else:
        active_dims = 'all'
        frozen_dims = 'none'
   
    if machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        output_folder = method_params['output_folder_x7']
        method_folder = method_params['method_folder_x7']
        with open("model_paths_x7.yaml") as tmp_file:
            if nnbatchid == -1:
                network_path = yaml.load(tmp_file)[method]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
            else:
                network_path = yaml.load(tmp_file)[method + '_batch'][nnbatchid]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

            print('Loading network from: ')
            print(network_path)
            # model = load_model(network_path + 'model_final.h5', custom_objects = {"huber_loss": tf.losses.huber_loss})

    if machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        output_folder = method_params['output_folder']
        method_folder = method_params['method_folder']
        with open("model_paths.yaml") as tmp_file:
            if nnbatchid == -1:
                network_path = yaml.load(tmp_file)[method]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

            else:
                network_path = yaml.load(tmp_file)[method + '_batch'][nnbatchid]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
                
            print('Loading network from: ')
            print(network_path)
            
            keras_model = keras.models.load_model(network_path + '/model_final.h5', compile = False)
            
    if machine == 'home':
        method_params = pickle.load(open("/users/afengler/OneDrive/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        output_folder = method_params['output_folder_home']
        method_folder = method_params['method_folder_home']
        with open("model_paths.yaml") as tmp_file:
            if nnbatchid == -1:
                network_path = yaml.load(tmp_file)[method]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

            else:
                network_path = yaml.load(tmp_file)[method + '_batch'][nnbatchid]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
                
            print('Loading network from: ')
            print(network_path)
            
    if data_type == 'perturbation_experiment':
        file_ = 'base_data_perturbation_experiment_nexp_1_n_' + str(n_samples) + '_' + infile_id + '.pickle'
        out_file_signature = 'post_samp_perturbation_experiment_nexp_1_n_' + str(n_samples) + '_' + infile_id                                                                      
    
    if data_type == 'parameter_recovery':
        file_ = 'parameter_recovery_data_binned_0_nbins_0_n_' + str(n_samples) + '/' + method + '_nchoices_2_parameter_recovery_binned_0_nbins_0_nreps_1_n_' + str(n_samples) + '.pickle'
        
        if analytic:
            pass
        else:  
            if not os.path.exists(output_folder + network_id):
                os.makedirs(output_folder + network_id)
        
        out_file_signature = 'post_samp_data_param_recov_unif_reps_1_n_' + str(n_samples) + '_' + infile_id
    
    if data_type == 'real':                                                                        
        file_ = args.infileid
        if machine == 'x7':
            data_folder = '/media/data_cifs/afengler/data/real/'
        if machine == 'ccv':
            data_folder = '/users/afengler/data/real/'
     
    method_params['n_choices'] = args.nchoices
    print('METHOD PARAMETERS: \n')
    print(method_params)

    # Load weights, biases and activations of current network --------
    if analytic:
        pass
    else:
        with open(network_path + "weights.pickle", "rb") as tmp_file:
            weights = pickle.load(tmp_file)
            #print(weights)
            for weight in weights:
                print(weight.shape)
        with open(network_path + 'biases.pickle', 'rb') as tmp_file:
            biases = pickle.load(tmp_file)
            #print(biases)
        with open(network_path + 'activations.pickle', 'rb') as tmp_file:
            activations = pickle.load(tmp_file)
            #print(activations)
        n_layers = int(len(weights))
# ----------------------------------------------------------------

# DEFINE TARGET LIKELIHOODS FOR CORRESPONDING MODELS -------------------------------------------------
 
# ----------------------------------------------------------------------------------------------------

# MAKE PARAMETER / DATA GRID -------------------------------------------------------------------------
    # REFORMULATE param bounds
    
    if data_type == 'real':
        print(data_folder + file_)
        data = pickle.load(open(data_folder + file_ , 'rb'))
        data_grid = data[0]
    elif data_type == 'parameter_recovery':
        data = pickle.load(open(method_folder + file_ , 'rb'))
        param_grid = data[0]
        data_grid = np.squeeze(data[1], axis = 0)

        # subset data according to array id so that we  
        data_grid = data_grid[((int(out_file_id) - 1) * n_by_arrayjob) : (int(out_file_id) * n_by_arrayjob), :, :]
        param_grid = param_grid[((int(out_file_id) - 1) * n_by_arrayjob) : (int(out_file_id) * n_by_arrayjob), :]

    elif data_type == 'perturbation_experiment':
        data = pickle.load(open(output_folder + file_ , 'rb'))
        param_grid = data[0]
        data_grid = data[1]
    else:
        print('Unknown Datatype, results will likely not make sense')   
    
    # 
    if args.samplerinit == 'random':
        init_grid = ['random' for i in range(data_grid.shape[0])]
    elif args.samplerinit == 'true':
        if not (data_type == 'parameter_recovery' or data_type == 'perturbation_experiment'):
            print('You cannot initialize true parameters if we are dealing with real data....')
        init_grid = data[0]
    elif args.samplerinit == 'mle':
        init_grid = ['mle' for i in range(data_grid.shape[0])]
    
    # Parameter bounds to pass to sampler    
    sampler_param_bounds = make_parameter_bounds_for_sampler(mode = mode, 
                                                             method_params = method_params)


    # Apply epsilon correction
    epsilon_bound_correction = 0.001
    sampler_param_bounds[:, 0] = sampler_param_bounds[:, 0] + epsilon_bound_correction
    sampler_param_bounds[:, 1] = sampler_param_bounds[:, 1] - epsilon_bound_correction

    sampler_param_bounds = [sampler_param_bounds for i in range(data_grid.shape[0])]
    
    print('sampler_params_bounds: ' , sampler_param_bounds)
    print('shape sampler param bounds: ', sampler_param_bounds[0].shape)
    print('active dims: ', active_dims)
    print('frozen_dims: ', frozen_dims)
    print('param_grid: ', param_grid)
    print('shape of param_grid:', len(param_grid))
    print('shape of data_grid:', data_grid.shape)
# ----------------------------------------------------------------------------------------------------

# RUN POSTERIOR SIMULATIONS --------------------------------------------------------------------------
   # MLP TARGET
    n_params = sampler_param_bounds[0].shape[0]
    
    if not analytic:
        mlpt = ktnpc.mlp_target(weights = weights, biases = biases, activations = activations, n_datapoints = data_grid.shape[1])

    def mlp_target(params, 
                   data, 
                   ll_min = -16.11809 # corresponds to 1e-7
                   ): 
        
        mlp_input_batch = np.zeros((data_grid.shape[1], sampler_param_bounds[0].shape[0] + 2), dtype = np.float32)
        mlp_input_batch[:, :n_params] = params
        mlp_input_batch[:, n_params:] = data

        #print(mlpt.predict(x = mlp_input_batch))
        # params_rep = np.tile(params, (data.shape[0], 1))
        # input_batch = np.concatenate([params_rep, data], axis = 1)
        #return np.sum(np.maximum(mlpt.predict(mlp_input_batch), ll_min))
        #return np.sum(np.core.umath.maximum(ktnp.predict(mlp_input_batch, weights, biases, activations, n_layers), ll_min))
        return np.sum(np.core.umath.maximum(keras_model(mlp_input_batch), ll_min))

    # def mlp_target(params,
    #                data,
    #                ll_min = -16.11809):
    #     mlp_input_batch[:, :n_params] = params
    #     mlp_input_batch[:, n_params:] = data
    #     #print(model.predict(mlp_input_batch).shape)
    #     return np.sum(np.maximum(model.predict(mlp_input_batch)[:, 0], ll_min))
    
    # NAVARRO FUSS (DDM)
    def nf_target(params, data, likelihood_min = 1e-10):
        return np.sum(np.maximum(np.log(batch_fptd(data[:, 0] * data[:, 1] * (- 1),
                                                   params[0],
                                                   params[1] * 2, 
                                                   params[2],
                                                   params[3])),
                                                   np.log(likelihood_min)))

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
        if sampler == 'slice':
            model = SliceSampler(bounds = args[2], 
                                 target = mlp_target, 
                                 w = .4 / 1024, 
                                 p = 8)

        if sampler == 'diffevo':
            model = DifferentialEvolutionSequential(bounds = args[2],
                                                    target = mlp_target,
                                                    mode_switch_p = 0.1,
                                                    gamma = 'auto',
                                                    crp = 0.3)
        
        (samples, lps, gelman_rubin_r_hat, random_seed) = model.sample(data = args[0],
                                                                       num_samples = n_slice_samples,
                                                                       init = args[1],
                                                                       active_dims = active_dims,
                                                                       frozen_dim_vals = frozen_dims)
        return (samples, lps, gelman_rubin_r_hat, random_seed)

    # Test navarro-fuss
    def nf_posterior(args): # TODO add active and frozen dim vals
        scp.random.seed()
        if sampler == 'slice':
            model = SliceSampler(bounds = args[2], 
                                 target = nf_target, 
                                 w = .4 / 1024, 
                                 p = 8)

        if sampler == 'diffevo':
            model = DifferentialEvolutionSequential(bounds = args[2],
                                                    target = nf_target,
                                                    mode_switch_p = 0.1,
                                                    gamma = 'auto',
                                                    crp = 0.3)
        
        (samples, lps, gelman_rubin_r_hat, random_seed) = model.sample(data = args[0],
                                                                       num_samples = n_slice_samples,
                                                                       init = args[1],
                                                                       active_dims = active_dims,
                                                                       frozen_dim_vals = frozen_dims)
       
        return (samples, lps, gelman_rubin_r_hat, random_seed)

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

    #print(data_grid.shape)
    #print(param_grid)
    #print(sampler_param_bounds)

    # Subset parameter and data grid
    
    # Run the sampler with correct target as specified above
    start_time = time.time()
    if n_cpus != 1:
        if method == 'lba_analytic':
            posterior_samples = np.array(p.map(lba_posterior, zip(data_grid,
                                                                  init_grid,
                                                                  sampler_param_bounds)))
        elif analytic and method == 'ddm_analytic':
            posterior_samples = p.map(nf_posterior, zip(data_grid, 
                                                        init_grid,
                                                        sampler_param_bounds))
        else:
            posterior_samples = p.map(mlp_posterior, zip(data_grid, init_grid, 
                                                         sampler_param_bounds))
    else:
        for i in range((out_file_id - 1) * 6, (out_file_id) * 6, 1):
            posterior_samples = mlp_posterior((data_grid[i],
                                               init_grid[i],
                                               sampler_param_bounds[i]))
    end_time = time.time()
    exec_time = end_time - start_time
    
    # Store files
    print('saving to file')
    if analytic:
        pickle.dump((param_grid, data_grid, posterior_samples, exec_time),
                    open(output_folder + out_file_signature + '_' + out_file_id + '.pickle', 'wb'))
        print(output_folder +  out_file_signature + '_' + out_file_id + ".pickle")

    else:
        print(output_folder + network_id + out_file_signature + '_' + out_file_id + ".pickle")
        pickle.dump((param_grid, data_grid, posterior_samples, exec_time), 
                    open(output_folder + network_id + out_file_signature + '_' + out_file_id + ".pickle", "wb"))