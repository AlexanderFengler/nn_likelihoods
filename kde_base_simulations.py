# Basic python utilities
import numpy as np
import scipy as scp
from scipy.stats import gamma

# Parallelization
import multiprocessing as mp
from  multiprocessing import Process
from  multiprocessing import Pool
import psutil

# System utilities
from datetime import datetime
import time
import os
import pickle
import uuid
import sys

# My own code
import kde_class as kde
import cddm_data_simulation as ds
import boundary_functions as bf

def bin_simulator_output(out = [0, 0],
                         bin_dt = 0.04,
                         n_bins = 0,
                         eps_correction = 1e-7,
                         params = ['v', 'a', 'w', 'ndt']
                        ): # ['v', 'a', 'w', 'ndt', 'angle']

    # hardcode 'max_t' to 20sec for now
    #n_bins = int(20.0 / bin_dt)
    if n_bins == 0:
        n_bins = int(out[2]['max_t'] / bin_dt)
        bins = np.linspace(0, out[2]['max_t'], n_bins + 1)
    else:    
        bins = np.linspace(0, out[2]['max_t'], n_bins + 1)
    
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

    labels = np.concatenate(counts, axis = 1)
    features = [out[2]['v'], out[2]['a'], out[2]['w'], out[2]['ndt']] #out[2]['theta']]
    return (features, labels, {'max_t': out[2]['max_t'], 
                               'bin_dt': bin_dt, 
                               'n_samples': out[2]['n_samples']})

def data_generator_ddm(*args):  # CAN I MAKE CONTEXT DEPENDENT???
    # CHOOSE SIMULATOR HERE
    simulator_data = ds.ddm_flexbound(*args)
    
    # CHOOSE TARGET DIRECTORY HERE
    #file_dir = '/users/afengler/data/kde/weibull_cdf/base_simulations_ndt_20000/'
    
    # USE FOR x7 MACHINE 
    #file_dir = '/media/data_cifs/afengler/tmp/'

    # STORE
    #file_name = file_dir + simulator + '_' + uuid.uuid1().hex
    #pickle.dump(simulator_data, open( file_name + '.pickle', "wb" ) )
    print(args)
    return simulator_data
    
if __name__ == "__main__":
    
    # INITIALIZATIONS ----------------------------------------------------------------------------------------
    # Get cpu cnt
    n_cpus = psutil.cpu_count(logical = False)
    machine = 'ccv'
    
    # Choose simulator and datatype
    method = 'weibull_cdf_ndt'
    binned = False
    
    # out file name components
    file_id = sys.argv[1]
    file_signature =  method + '_base_simulations_'
    
    # Load meta data from kde_info.pickle file
    if machine == 'x7':
        method_folder = "/media/data_cifs/afengler/data/kde/ddm/"

    if machine == 'ccv':
        method_folder = '/users/afengler/data/kde/' + method + '/'

    if machine == 'x7':
        stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        method_params = stats[method]
    if machine == 'ccv':
        stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        method_params = stats[method]

    out_folder = method_folder + 'base_simulations_ndt_20000/'
    

    
    # Simulator parameters
    s = 1 # Choose
    delta_t = 0.01 # Choose
    max_t = 20 # Choose
    n_samples = 1000 # Choose
    n_simulators = 10000 # Choose
    print_info = False # Choose
    bound = method_params['boundary']
    boundary_multiplicative = method_params['boundary_multiplicative'] 
    
    # Extra params
    bin_dt = 0.04
    n_bins = 256
    # --------------------------------------------------------------------------------------------------------
    
    # GENERATE A SET OF PARAMETERS ---------------------------------------------------------------------------
    process_param_names = method_params['param_names']
    boundary_param_names = method_params['boundary_param_names']
    param_names_full = process_param_names + boundary_param_names
    process_param_upper_bnd = []
    process_param_lower_bnd = []
    
    for i in range(len(process_param_names)):
        process_param_upper_bnd.append(method_params['param_bounds_network'][i][0]) 
        process_param_lower_bnd.append(method_params['param_bounds_network'][i][1])
        
    param_samples = tuple(map(tuple, np.random.uniform(low = process_param_lower_bnd,
                                      high = process_param_upper_bnd,
                                      size = (n_simulators, len(process_param_names)))))

    if len(boundary_param_names) > 0:
        boundary_param_lower_bnd = []
        boundary_param_upper_bnd = []
        
        for i in range(len(boundary_param_names)):
            boundary_param_lower_bnd.append(method_params['boundary_param_bounds_network'][i][0])
            boundary_param_upper_bnd.append(method_params['boundary_param_bounds_network'][i][0])
                                      
        boundary_param_samples = np.random.uniform(low = boundary_param_lower_bnd,
                                                   high = bonudary_param_upper_bnd,
                                                   size = (n_simulators, len(boundary_param_names)))
                                      

    # --------------------------------------------------------------------------------------------------------
    
    # DEFINE FUNCTIONS THAAT NEED INITIALIZATION DEPENDEND ON CONTEXT ----------------------------------------
    def data_generator_ddm_binned(*args):
        simulator_data = ds.ddm_flexbound(*args)
        features, labels, meta = bin_simulator_output(out = simulator_data,
                                                      bin_dt = bin_dat, 
                                                      n_bins = n_bins,
                                                      eps_correction = 1e-7,
                                                      params = param_names_full) 
        return (features, labels, meta) 
    # --------------------------------------------------------------------------------------------------------
    
    # MAKE SUITABLE FOR PARALLEL SIMULATION ------------------------------------------------------------------
    args_list = []
    for i in range(n_simulators):
        # Get current set of parameters
        process_params = param_samples[i]
        #process_params = (v_sample[i], a_sample[i], w_sample[i], ndt_sample[i], s)
        sampler_params = (delta_t, max_t, n_samples, print_info, bound, boundary_multiplicative)
                          
        if len(boundary_param_names) > 0:
            boundary_params = (dict(zip(boundary_param_names, boundary_param_samples[i])) ,)
        else:
            boundary_params = ({},)
        
        # Append argument list with current parameters
        args_tmp = process_params + sampler_params + boundary_params
        args_list.append(args_tmp)
    # --------------------------------------------------------------------------------------------------------
                   
    # RUN SIMULATIONS AND STORE DATA -------------------------------------------------------------------------
    # BINNED VERSION
    if binned:
        # Parallel Loop
        with Pool(processes = n_cpus) as pool:
            res = pool.starmap(data_generator_ddm_binned, args_list)

        features = []
        labels = []
        for i in range(len(res)):
            features.append(res[i][1])
            labels.append(res[i][0])

        features = np.array(features)
        labels = np.array(labels)
        meta = res[0][2]
        
        # Storing files
        pickle.dump((features, labels), open(out_folder + file_signature + file_id + '.pickle', 'wb'))
        pickle.dump(meta, open(out_folder + 'meta_' + file_signature + '.pickle', 'wb'))
    
    # STANDARD OUTPUT
    else:
        # Parallel Loop
        with Pool(processes = n_cpus) as pool:
            res = pool.starmap(data_generator_ddm, args_list)
            pickle.dump(res, open(out_folder + file_signature + file_id + '.pickle', 'wb'))
    # --------------------------------------------------------------------------------------------------------
    print('finished')


# UNUSED ---------------------------------------------------------------

    # Parameter ranges (for the simulator)
#     v = [-2, 2]
#     w = [0.3, 0.7]
#     a = [0.5, 2]
#     g = [-1.0, 1.0]
#     b = [-1.0, 1.0]

    # DDM
#     v = [-2.0, 2.0]
#     a = [0.5, 2.0]
#     w = [0.3, 0.7]
#     ndt = [0, 1]
    
    # Bound - Angle
#     theta = [0, np.pi/2 - 0.2]

    # LCA
#     v = [-2.0, 2.0]
#     w = [0.3, 0.7]
#     a = [0.5, 2.0]
#     g = [-1.0, 0.4]
#     b = [-1.0, 0.4]
    
    
    # FULL DDM
#     dw = [0.0, 0.1]
#     sdv = [0.0, 0.5]

    #     c1 = [0, 5]
#     c2 = [1, 1.5]
    # Linear Collapse
#     node = [0, 2]
#     theta = [0, np.pi/2 - 0.2]

    # Weibull Bound
#     node = [0, 5]
#     shape = [1.1, 50]
#     scale = [0.1, 10]


# Make function input tuples
    # DDM
#     v_sample = np.random.uniform(low = v[0], high = v[1], size = n_simulators)
#     w_sample = np.random.uniform(low = w[0], high = w[1], size = n_simulators)
#     a_sample = np.random.uniform(low = a[0], high = a[1], size = n_simulators)
#     ndt_sample = np.random.uniform(low = ndt[0], high = ndt[1], size = n_simulators)
    
    # BOUND - ANGLE
#     theta_sample = np.random.uniform(low = theta[0], high = theta[1], size = n_simulators) 
    
    # BOUND - ANGLE
    
    
    # Ornstein 
#     g_sample = np.random.uniform(low = g[0], high = g[1], size = n_simulators)
    
    # LCA
#     n_particles = 2
#     v_sample = []
#     w_sample = []
    
#     for i in range(n_simulators):
#         v_tmp = np.random.uniform(low = v[0], high = v[1])
#         w_tmp = np.random.uniform(low = w[0], high = w[1])
        
#         v_sample.append(np.array([v_tmp] * n_particles, dtype = np.float32))
#         w_sample.append(np.array([w_tmp] * n_particles, dtype = np.float32))
        
#     a_sample = np.random.uniform(low = a[0], high = a[1], size = n_simulators)
#     g_sample = np.random.uniform(low = g[0], high = g[1], size = n_simulators)
#     b_sample = np.random.uniform(low = b[0], high = b[1], size = n_simulators)
#     s = np.array([1] * n_particles)

    # Full DDM
#     dw_sample = np.random.uniform(low = dw[0], high = dw[1], size = n_simulators)
#     sdv_sample = np.random.uniform(low = sdv[0], high = sdv[1], size = n_simulators)
    
    # Exp c1_c2
#     c1_sample = np.random.uniform(low = c1[0], high = c1[1], size = n_simulators)
#     c2_sample = np.random.uniform(low = c2[0], high = c2[1], size = n_simulators)

    # Linear Collapse
#     node_sample = np.random.uniform(low = node[0], high = node[1], size = n_simulators)
#     theta_sample = np.random.uniform(low = theta[0], high = theta[1], size = n_simulators)

    # Weibull
#     node_sample = np.random.uniform(low = node[0], high = node[1], size = n_simulators)
#     shape_sample = np.random.uniform(low = shape[0], high = shape[1], size = n_simulators)
#     scale_sample = np.random.uniform(low = scale[0], high = scale[1], size = n_simulators)

    # Defining main function to iterate over:
    # Folder in which we would like to dump files