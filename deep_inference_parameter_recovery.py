import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
import numpy as np
import yaml
import pandas as pd
from itertools import product
import pickle
import uuid
import os

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from cdwiener import batch_fptd
import clba

# INITIALIZATIONS -------------------------------------------------------------
#print(device_lib.list_local_devices())

machine = 'x7'
method = 'ddm_ndt'
#param_origin = 'previous'

#analytic = True
#file_signature = '_start_true_'
#n_data_samples = 2500  # should be hardwired to 2500 for deep inference
#n_sims = 500
fcn_custom_objects = {"heteroscedastic_loss": tf.losses.huber_loss}

# Get list of files
method_comparison_folder = "/media/data_cifs/afengler/data/kde/ddm/method_comparison/"
file_signature  = 'kde_sim_test_ndt_'
file_signature_len = len(file_signature)
files = os.listdir(method_comparison_folder)

# Get data in desired format
dats = []
for file_ in files:
    if file_[:file_signature_len] == file_signature:
        dats.append(pickle.load(open(method_comparison_folder + file_ , 'rb')))


dat_tmp_0 = []
dat_tmp_1 = []
for dat in dats:
    dat_tmp_0.append(dat[0])
    dat_tmp_1.append(dat[1])

dat_total = [np.concatenate(dat_tmp_0, axis = 0), np.concatenate(dat_tmp_1, axis = 0)]
data_grid = dat_total[1]
param_grid = dat_total[0]
    
# Get network hyperparameters
dnn_params = yaml.load(open("hyperparameters.yaml"))
print(' GPU I AM ASKING FOR: ', dnn_params['gpu_x7'])
if machine == 'x7':
    stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    output_folder = method_params['output_folder_x7']
    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = dnn_params['gpu_x7'] 

    with open("model_paths_x7.yaml") as tmp_file:
        network_path = yaml.load(tmp_file)['fcn_' + method]

if machine == 'ccv':
    stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    output_folder = method_params['output_folder']
    with open("model_paths.yaml") as tmp_file:
        network_path = yaml.load(tmp_file)['fcn_' + method]
        
#print(stats)
#print(method_params)

# MAKE PARAMETER / DATA GRID -------------------------------------------------------------------------

fcn = keras.models.load_model(network_path + 'model_final.h5', custom_objects = fcn_custom_objects)
fcn_results = fcn.predict(data_grid)
pickle.dump((param_grid, data_grid, fcn_results), open(output_folder + "deep_inference_sim_test_ndt_{}.pickle".format(uuid.uuid1()), "wb"))

# # REFORMULATE param bounds
# def generate_param_grid():
#     param_upper_bnd = []
#     param_lower_bnd = []
#     boundary_param_upper_bnd = [] 
#     boundary_param_lower_bnd = []

#     for p in range(len(method_params['param_names'])):
#         param_upper_bnd.append(method_params['param_bounds_sampler'][p][1])
#         param_lower_bnd.append(method_params['param_bounds_sampler'][p][0])

#     if len(method_params['boundary_param_names']) > 0:
#         for p in range(len(method_params['boundary_param_names'])):
#             boundary_param_upper_bnd.append(method_params['boundary_param_bounds'][p][1])
#             boundary_param_lower_bnd.append(method_params['boundary_param_bounds'][p][0])                                    

#     param_grid = np.random.uniform(low = param_lower_bnd, 
#                                    high = param_upper_bnd, 
#                                    size = (n_sims, len(method_params['param_names'])))

#     if len(method_params['boundary_param_names']) > 0:
#         boundary_param_grid = np.random.uniform(low = boundary_param_lower_bnd,
#                                                 high = boundary_param_upper_bnd,
#                                                 size = (n_sims, len(method_params['boundary_param_bounds'])))
#     else:
#         boundary_param_grid = []
        
#     return (param_grid, boundary_param_grid)

# # REFORMULATE param bounds
# def generate_param_grid_lba2():
#     param_upper_bnd = []
#     param_lower_bnd = []
#     boundary_param_upper_bnd = [] 
#     boundary_param_lower_bnd = []

#     for p in range(len(method_params['param_names'])):
#         param_upper_bnd.append(method_params['param_bounds_sampler'][p][1])
#         param_lower_bnd.append(method_params['param_bounds_sampler'][p][0])

#     if len(method_params['boundary_param_names']) > 0:
#         for p in range(len(method_params['boundary_param_names'])):
#             boundary_param_upper_bnd.append(method_params['boundary_param_bounds'][p][1])
#             boundary_param_lower_bnd.append(method_params['boundary_param_bounds'][p][0])                                    

#     param_grid = np.random.uniform(low = param_lower_bnd, 
#                                    high = param_upper_bnd, 
#                                    size = (n_sims, len(method_params['param_names'])))
    
#     # Adjust v_1 so that we are unlikely to get not observations for either choice
#     # Works only for two choices
#     param_grid[:, 1] = param_grid[:, 0] + (param_grid[:, 4] * np.random.uniform(low = - 2.0, high = 2.0, size = n_sims))

#     if len(method_params['boundary_param_names']) > 0:
#         boundary_param_grid = np.random.uniform(low = boundary_param_lower_bnd,
#                                                 high = boundary_param_upper_bnd,
#                                                 size = (n_sims, len(method_params['boundary_param_bounds'])))
#     else:
#         boundary_param_grid = []
        
#     return param_grid
                     
# def generate_data_grid(param_grid, boundary_param_grid):
#     data_grid = np.zeros((n_sims, n_data_samples, 2))
#     for i in range(n_sims):
#         param_dict_tmp = dict(zip(method_params["param_names"], param_grid[i]))
        
#         if len(method_params['boundary_param_names']) > 0:
#             boundary_dict_tmp = dict(zip(method_params["boundary_param_names"], boundary_param_grid[i]))
#         else:
#             boundary_dict_tmp = {}
            
#         rts, choices, _ = method_params["dgp"](**param_dict_tmp, 
#                                                boundary_fun = method_params["boundary"], 
#                                                n_samples = n_data_samples,
#                                                delta_t = 0.01, 
#                                                boundary_params = boundary_dict_tmp,
#                                                boundary_multiplicative = method_params['boundary_multiplicative'])
        
#         data_grid[i] = np.concatenate([rts, choices], axis = 1)
#     return data_grid

# def generate_data_grid_lba2(param_grid):
#     data_grid = np.zeros((n_sims, n_data_samples, 2))
#     param_names_tmp = ['v', 'A', 'b', 's', 'ndt']
#     for i in range(n_sims):
#         params_tmp = []
#         params_tmp.append(np.array(param_grid[i][:2]))
#         params_tmp.append(np.array(param_grid[i][2]))
#         params_tmp.append(np.array(param_grid[i][3]))
#         params_tmp.append(np.array(param_grid[i][4])) 
#         params_tmp.append(np.array(param_grid[i][5]))
#         params_dict_tmp = dict(zip(param_names_tmp, params_tmp))
#         print('params_dict: ', params_dict_tmp)
#         # Generate data
#         rts, choices, _ = method_params["dgp"](**params_dict_tmp,
#                                                n_samples = n_data_samples)
#         data_grid[i] = np.concatenate([rts, choices], axis = 1)
#     return data_grid


# if method[:3] == 'lba':
#     param_grid = generate_param_grid_lba2()
#     data_grid = generate_data_grid_lba2(param_grid) 
# else:   
#     param_grid, boundary_param_grid = generate_param_grid() 
#     data_grid = generate_data_grid(param_grid, boundary_param_grid)
#     if len(method_params['boundary_param_names']) > 0:
#         param_grid = np.concatenate([param_grid, boundary_param_grid], axis = 1)

# print('param_grid: ', param_grid)
# print('shape of data_grid:', data_grid.shape)

# Get data from other sampling run
