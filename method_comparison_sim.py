# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import yaml
import pandas as pd
from itertools import product
from samplers import SliceSampler
#from slice_sampler import SliceSampler
import pickle
import uuid

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from cdwiener import batch_fptd
#from np_network import np_predict
#from kde_info import KDEStats

import keras_to_numpy as ktnp

# INITIALIZATIONS -------------------------------------------------------------
machine = 'ccv'
method = 'ddm'
n_data_samples = 2000
n_slice_samples = 5000
n_sims = 10
n_cpus = 'all'

stats = pickle.load(open("kde_stats.pickle", "rb"))
method_params = stats[method]

print(stats)
print(method_params)

if machine == 'x7':
    output_folder = method_params['output_folder_x7']
    with open("model_paths_x7.yaml") as tmp_file:
        network_path = yaml.load(tmp_file)[method]
if machine == 'ccv':
    output_folder = method_params['output_folder']
    with open("model_paths.yaml") as tmp_file:
        network_path = yaml.load(tmp_file)[method]

# model = keras.models.load_model(network_path, custom_objects=custom_objects)
# fcn = keras.models.load_model(fcn_path, custom_objects=fcn_custom_objects)

# Load weights, biases and activations of current network --------
with open(network_path + "weights.pickle", "rb") as tmp_file:
    weights = pickle.load(tmp_file)
with open(network_path + 'biases.pickle', 'rb') as tmp_file:
    biases = pickle.load(tmp_file)
with open(network_path + 'activations.pickle', 'rb') as tmp_file:
    activations = pickle.load(tmp_file)
# ----------------------------------------------------------------
def target(params, data, ll_min = 1e-29, ndt = True):
    if ndt == False:
        params_rep = np.tile(params, (data.shape[0], 1))
        input_batch = np.concatenate([params_rep, data], axis = 1)
        out = ktnp.predict(input_batch, weights, biases, activations)
        return np.sum(out)
    else:
        params_rep = np.tile(params[:-1], (data.shape[0], 1))
        data[0,:] = data[:, 0] - params[-1]
        input_batch = np.concatenate([params_rep, data], axis = 1)
        out = ktnp.predict(input_batch, weights, biases, activations)
        out[data[:, 0] <= 0] = np.log(ll_min)
        return np.sum(out)

def nf_target(params, data):
    return np.log(batch_fptd(data[:, 0] * data[:, 1] * (-1), params[0],
         params[1] * 2, params[2])).sum()

# MAKE PARAMETER / DATA GRID -------------------------------------------------------------------------

# REFORMULATE param bounds
def generate_param_grid():
    param_upper_bnd = []
    param_lower_bnd = []
    boundary_param_upper_bnd = [] 
    boundary_param_lower_bnd = []

    for p in range(len(method_params['param_names'])):
        param_upper_bnd.append(method_params['param_bounds'][p][1])
        param_lower_bnd.append(method_params['param_bounds'][p][0])

    if len(method_params['boundary_param_names']) > 0:
        for p in range(len(method_params['boundary_param_names'])):
            boundary_param_upper_bnd.append(method_params['boundary_param_bounds'][p][1])
            boundary_param_lower_bnd.append(method_params['boundary_param_bounds'][p][0])                                    

    param_grid = np.random.uniform(low = param_lower_bnd, 
                                   high = param_upper_bnd, 
                                   size = (n_sims, len(method_params['param_names'])))

    if len(method_params['boundary_param_names']) > 0:
        boundary_param_grid = np.random.uniform(low = boundary_param_lower_bnd,
                                                high = boundary_param_upper_bnd,
                                                size = (n_sims, len(method_params['boundary_param_bounds'])))
    else:
        boundary_param_grid = []
        
    return (param_grid, boundary_param_grid)
                     
def generate_data_grid(param_grid, boundary_param_grid):
    data_grid = np.zeros((n_sims, n_data_samples, 2))
    for i in range(n_sims):
        param_dict_tmp = dict(zip(method_params["param_names"], param_grid[i]))
        
        if len(method_params['boundary_param_names']) > 0:
            boundary_dict_tmp = dict(zip(method_params["boundary_param_names"], boundary_param_grid[i]))
        else:
            boundary_dict_tmp = {}
            
        rts, choices, _ = method_params["dgp"](**param_dict_tmp, 
                                               boundary_fun = method_params["boundary"], 
                                               n_samples = n_data_samples,
                                               delta_t = 0.01, 
                                               boundary_params = boundary_dict_tmp)
        
        data_grid[i] = np.concatenate([rts, choices], axis = 1)
    return data_grid

def generate_data_grid_lba(param_grid):
    data_grid = np.zeros((n_sims, n_data_samples, 2))
    param_names_tmp = ['v', 'A', 'b', 's']
    for i in range(n_sims):
        params_tmp = []
        params_tmp.append(np.array(param_grid[i][:(len(param_grid[i]) - 3)]))
        params_tmp.append(np.array(param_grid[i][len(param_grid[i]) - 3]))
        params_tmp.append(np.array(param_grid[i][len(param_grid[i]) - 2]))
        params_tmp.append(np.array(param_grid[i][len(param_grid[i]) - 1]))     
        params_dict_tmp = dict(zip(param_names_tmp, params_tmp))
        print(params_dict_tmp['v'])
        # Generate data
        rts, choices, _ = method_params["dgp"](**params_dict_tmp,
                                               n_samples = n_data_samples)
        data_grid[i] = np.concatenate([rts, choices], axis = 1)
    return data_grid

param_grid, boundary_param_grid = generate_param_grid() 
print('param_grid', param_grid[0])
if method[:3] == 'lba':
    data_grid = generate_data_grid_lba(param_grid) 
else:   
    data_grid = generate_data_grid(param_grid, boundary_param_grid)

print(data_grid)
print(data_grid.shape)
# ----------------------------------------------------------------------------------------------------

# RUN POSTERIOR SIMULATIONS --------------------------------------------------------------------------

sampler_param_bounds = np.array(method_params["param_bounds"] + method_params["boundary_param_bounds"])

def kde_posterior(data):
    model = SliceSampler(bounds = sampler_param_bounds, 
                         target = target, 
                         w = .4 / 1024, 
                         p = 8)
    model.sample(data, num_samples = n_slice_samples)
    return model.samples

#test navarro-fuss
def nf_posterior(data):
    model = SliceSampler(bounds = sampler_param_bounds,
                         target = nf_target, 
                         w = .4 / 1024, 
                         p = 8)
    model.sample(data, num_samples = n_slice_samples)
    return model.samples

if n_cpus == 'all':
    p = mp.Pool(mp.cpu_count())
    
else: 
    p = mp.Pool(n_cpus)

kde_results = np.array(p.map(kde_posterior, data_grid))

#print(target([0, 1.5, 0.5], data_grid[0]))
# import ipdb; ipdb.set_trace()
# kde_results = kde_posterior(data_grid[0])

# if method == "ddm":
#     nf_results = p.map(test_nf, data_grid)

# print("nf finished!")

# fcn_results = fcn.predict(data_grid)

# print("fcn finished!")

pickle.dump((param_grid, data_grid, kde_results), open(output_folder + "kde_sim_test_ndt_{}.pickle".format(uuid.uuid1()), "wb"))

# pickle.dump((param_grid, fcn_results), open(output_folder + "fcn_sim_random{}.pickle".format(part), "wb"))

# if method == "ddm":
#     pickle.dump((param_grid, data_grid, nf_results), open(output_folder + "nf_sim_{}.pickle".format(uuid.uuid1()), "wb"))
# ------------------------------------------------------------------------------------------------------
    
# UNUSED ---------
# n_sims_per_param = 10
# n_sims = n_sims_per_param ** len(param_names)

# data_grid = np.zeros((n_sims, n_data_samples, 2))
# data_grid = data_grid[(data_grid.shape[0] // 2):]
# data_grid = data_grid[(part * data_grid.shape[0] // 4):((part + 1) * data_grid.shape[0] // 4)]

# param_grid = np.linspace(param_bounds[0], param_bounds[1], num=n_sims_per_param)
# param_grid = np.array(list(product(*param_grid.T)))
# param_grid = param_grid[(part * param_grid.shape[0] // 4):((part + 1) * param_grid.shape[0] // 4)]