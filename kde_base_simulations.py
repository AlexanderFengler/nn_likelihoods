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
                         eps_correction = 1e-7,
                         params = ['v', 'a', 'w', 'ndt', 'angle']):

    # hardcode 'max_t' to 20sec for now
    #n_bins = int(20.0 / bin_dt + 1)
    n_bins = int(out[2]['max_t'] / bin_dt + 1)
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

    labels = np.concatenate(counts, axis = 1)
    features = [out[2]['v'], out[2]['a'], out[2]['w'], out[2]['ndt'], out[2]['theta']]
    return (features, labels)

def data_generator(*args):
    # CHOOSE SIMULATOR HERE
    simulator_data = ds.ddm_flexbound(*args)
    
    # CHOOSE TARGET DIRECTORY HERE
    file_dir = '/users/afengler/data/kde/angle/base_simulations_ndt_20000/'
    
    # USE FOR x7 MACHINE 
    #file_dir = '/media/data_cifs/afengler/tmp/'

    # STORE
    file_name = file_dir + simulator + '_' + uuid.uuid1().hex
    pickle.dump(simulator_data, open( file_name + '.pickle', "wb" ) )
    print(args)

    
    
def data_generator_binned(*args):
    simulator_data = ds.ddm_flexbound(*args)
    #file_dir = '/users/afengler/data/kde/angle/base_simulations_ndt_20000/'
    features, labels = bin_simulator_output(out = simulator_data,
                                            bin_dt = 0.04, 
                                            eps_correction = 1e-7,
                                            params = ['v','a', 'w', 'ndt', 'theta'])
    return (features, labels)     
    
if __name__ == "__main__":
    # Get cpu cnt
    n_cpus = psutil.cpu_count(logical = False)

    # Parameter ranges (for the simulator)
#     v = [-2, 2]
#     w = [0.3, 0.7]
#     a = [0.5, 2]
#     g = [-1.0, 1.0]
#     b = [-1.0, 1.0]
    
    # DDM
    v = [-2.0, 2.0]
    a = [0.5, 2.0]
    w = [0.3, 0.7]
    ndt = [0, 1]
    
    # Bound - Angle
    theta = [0, np.pi/2 - 0.2]

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

    # Simulator parameters
    simulator = 'ddm'
    s = 1
    delta_t = 0.01
    max_t = 10
    n_samples = 100000
    print_info = False
    bound = bf.angle # CHOOSE BOUNDARY FUNCTION
    boundary_multiplicative = False # CHOOSE WHETHER BOUNDARY IS MULTIPLICATIVE (W.R.T Starting separation) OR NOT

    # Number of simulators to run
    n_simulators = 1500

    # Make function input tuples
    # DDM
    v_sample = np.random.uniform(low = v[0], high = v[1], size = n_simulators)
    w_sample = np.random.uniform(low = w[0], high = w[1], size = n_simulators)
    a_sample = np.random.uniform(low = a[0], high = a[1], size = n_simulators)
    ndt_sample = np.random.uniform(low = ndt[0], high = ndt[1], size = n_simulators)
    
    # BOUND - ANGLE
    theta_sample = np.random.uniform(low = theta[0], high = theta[1], size = n_simulators) 
    
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

    args_list = []
    for i in range(n_simulators):
        # Get current set of parameters
        process_params = (v_sample[i], a_sample[i], w_sample[i], ndt_sample[i], s)
        sampler_params = (delta_t, max_t, n_samples, print_info, bound, boundary_multiplicative)
        # CHOOSE
        boundary_params = ({'theta': theta_sample[i]},)
        
        # Append argument list with current parameters
        args_tmp = process_params + sampler_params + boundary_params
        args_list.append(args_tmp)
    # Parallel Loop
    with Pool(processes = n_cpus) as pool:
        res = pool.starmap(data_generator_binned, args_list)
    
    features = []
    labels = []
    for i in range(len(res)):
        features.append(res[i][1])
        labels.append(res[i][0])
    
    features = np.array(features)
    labels = np.array(labels)
    
    
    
    pickle.dump((features, labels), open('/users/afengler/data/tmp/binned_data_test.pickle', 'wb'))
    print('finished')