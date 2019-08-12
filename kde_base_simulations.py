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

# My own code
import kde_class as kde
import cddm_data_simulation as ds
import boundary_functions as bf

def data_generator(*args):
    # CHOOSE SIMULATOR HERE
    simulator_data = ds.lca(*args)
    
    # CHOOSE TARGET DIRECTORY HERE
    file_dir =  '/users/afengler/data/kde/lca/base_simulations_20000/'

    # STORE
    file_name = file_dir + simulator + '_' + uuid.uuid1().hex
    pickle.dump(simulator_data, open( file_name + '.pickle', "wb" ) )
    print('success')

if __name__ == "__main__":
    # Get cpu cnt
    n_cpus = psutil.cpu_count(logical = False)

    # Parameter ranges (for the simulator)
#     v = [-2, 2]
#     w = [0.3, 0.7]
#     a = [0.5, 2]
#     g = [-1.0, 1.0]
#     b = [-1.0, 1.0]
    
    # LCA
    v = [-2.0, 2.0]
    w = [0.3, 0.7]
    a = [0.5, 2.0]
    g = [-1.0, 0.4]
    b = [-1.0, 0.4]
    
    
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
    simulator = 'lca'
    s = 1
    delta_t = 0.01
    max_t = 40
    n_samples = 20000
    print_info = False
    bound = bf.constant # CHOOSE BOUNDARY FUNCTION
    boundary_multiplicative = True # CHOOSE WHETHER BOUNDARY IS MULTIPLICATIVE (W.R.T Starting separation) OR NOT

    # Number of simulators to run
    n_simulators = 100000

    # Make function input tuples
#     v_sample = np.random_uniform(low = v[0], high = v[1], size = n_simulators)
#     w_sample = np.random.uniform(low = w[0], high = w[1], size = n_simulators)
#     a_sample = np.random.uniform(low = a[0], high = a[1], size = n_simulators)

    # Ornstein 
#     g_sample = np.random.uniform(low = g[0], high = g[1], size = n_simulators)
    
    # LCA
    n_particles = 2
    v_sample = []
    w_sample = []
    
    for i in range(n_simulators):
        v_tmp = np.random.uniform(low = v[0], high = v[1])
        w_tmp = np.random.uniform(low = w[0], high = w[1])
        
        v_sample.append(np.array([v_tmp] * n_particles))
        w_sample.append(np.array([w_tmp] * n_particles))
        
    a_sample = np.random.uniform(low = a[0], high = a[1], size = n_simulators)
    g_sample = np.random.uniform(low = g[0], high = g[1], size = n_simulators)
    b_sample = np.random.uniform(low = b[0], high = b[1], size = n_simulators)
    s = np.array([1] * n_particles)

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
        process_params = (v_sample[i], a_sample[i], w_sample[i], g_sample[i], b_sample[i], s)
        sampler_params = (delta_t, max_t, n_samples, print_info, bound, boundary_multiplicative)
        boundary_params = ({},)
        
        # Append argument list with current parameters
        args_tmp = process_params + sampler_params + boundary_params
        args_list.append(args_tmp)

    # Parallel Loop
    with Pool(processes = n_cpus) as pool:
        res = pool.starmap(data_generator, args_list)