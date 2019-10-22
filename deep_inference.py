from tensorflow import keras
import numpy as np
from cdwiener import array_fptd
import os
import pandas as pd
import re
import time
from datetime import datetime
import pickle
import yaml
import sys

def heteroscedastic_loss(true, pred):
    params = pred.shape[1] // 2
    point = pred[:, :params]
    var = pred[:, params:]
    precision = 1 / var
    return keras.backend.sum((precision * (true - point)) ** 2 + keras.backend.log(var), - 1)

def make_fcn(n_params = 5,
             input_dims = (None, 2),
             conv_layers = [64, 64, 128, 128, 128], 
             kernel_sizes = [1, 3, 3, 3, 3], 
             strides = [1, 2, 2, 2, 2], 
             activations = ["relu", "relu", "relu", "relu", "relu"]):
    
    # Input layer
    inp = keras.Input(shape = input_dims)
    x = inp
    for layer in range(len(conv_layers)):
        x = keras.layers.Conv1D(conv_layers[layer], 
                                kernel_size = kernel_sizes[layer], 
                                strides = strides[layer], 
                                activation = activations[layer])(x)
    # Pooling layer
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Final Layer 
    mean = keras.layers.Dense(n_params)(x)
    var = keras.layers.Dense(n_params, activation = "softplus")(x)
    out = keras.layers.Concatenate()([mean, var])
    model = keras.Model(inp, out)
    return model

# REFORMULATE param bounds
def generate_param_grid(n_datasets = 100):
    param_upper_bnd = []
    param_lower_bnd = []
    boundary_param_upper_bnd = [] 
    boundary_param_lower_bnd = []

    for p in range(len(method_params['param_names'])):
        param_upper_bnd.append(method_params['param_bounds_network'][p][1])
        param_lower_bnd.append(method_params['param_bounds_network'][p][0])

    if len(method_params['boundary_param_names']) > 0:
        for p in range(len(method_params['boundary_param_names'])):
            boundary_param_upper_bnd.append(method_params['boundary_param_bounds_network'][p][1])
            boundary_param_lower_bnd.append(method_params['boundary_param_bounds_network'][p][0])                                    

    param_grid = np.random.uniform(low = param_lower_bnd, 
                                   high = param_upper_bnd, 
                                   size = (n_datasets, len(method_params['param_names'])))

    if len(method_params['boundary_param_names']) > 0:
        boundary_param_grid = np.random.uniform(low = boundary_param_lower_bnd,
                                                high = boundary_param_upper_bnd,
                                                size = (n_datasets, len(method_params['boundary_param_bounds_network'])))
    else:
        boundary_param_grid = []
        
    return (param_grid, boundary_param_grid)

def generate_data_grid(param_grid, boundary_param_grid):
    data_grid = np.zeros((param_grid.shape[0], n_data_samples, 2))
    for i in range(param_grid.shape[0]):
        param_dict_tmp = dict(zip(method_params["param_names"], param_grid[i]))
        
        if len(method_params['boundary_param_names']) > 0:
            boundary_dict_tmp = dict(zip(method_params["boundary_param_names"], boundary_param_grid[i]))
        else:
            boundary_dict_tmp = {}
            
        rts, choices, _ = method_params["dgp"](**param_dict_tmp, 
                                               boundary_fun = method_params["boundary"], 
                                               n_samples = n_data_samples,
                                               delta_t = 0.01, 
                                               boundary_params = boundary_dict_tmp,
                                               boundary_multiplicative = method_params['boundary_multiplicative'])
        
        data_grid[i] = np.concatenate([rts, choices], axis = 1)
        
        if i % 100 == 0:
            print('datasets_generated: ', i)
    return data_grid
                  

#data_folder = "/users/afengler/data/kde/full_ddm/train_test_data_20000/"
#files = os.listdir(data_folder)
#files = [f for f in files if re.match("data_.*", f)]

timestamp = datetime.now().strftime('%m_%d_%y_%H_%M_%S')
# save model configuration and hyperparams in folder

# INITIALIZATIONS -------------------------------------------------------------
machine = 'x7'
method = 'ddm_ndt'
analytic = True
file_signature = '_start_true_'
n_data_samples = int(sys.argv[1])
n_datasets = 500000
n_cpus = 'all'

dnn_params = yaml.load(open("hyperparameters.yaml"))

# set up gpu to use
if machine == 'x7':
    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = dnn_params['gpu_x7'] 
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

if machine == 'x7':
    folder_str = "/media/data_cifs/afengler/"
if machine == 'ccv':
    folder_str = "/users/afengler/"

stats = pickle.load(open(folder_str + "git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
method_params = stats[method]

# Specify final model path
model_path_x7 = method_params['model_folder_x7'] + "deep_inference_" + str(int(n_data_samples)) + "_" + timestamp
model_path_ccv = method_params['model_folder'] + "deep_inference_" + str(int(n_data_samples)) + "_" + timestamp

if machine == 'x7':
    if not os.path.exists(model_path_x7):
        os.makedirs(model_path_x7)
if machine == 'ccv':
    if not os.path.exists(model_path_ccv):
        os.makedirs(model_path_ccv)

    
# Update model paths in model_path.yaml -----------------------------------------------------------------------
print(model_path_x7)
model_paths_x7 = yaml.load(open(folder_str + 'git_repos/nn_likelihoods/model_paths_x7.yaml'))
print(model_paths_x7)
model_paths_x7['fcn_' + method + '_' + str(int(n_data_samples))] = model_path_x7
yaml.dump(model_paths_x7, open(folder_str + 'git_repos/nn_likelihoods/model_paths_x7.yaml', "w"))

model_paths_ccv = yaml.load(open(folder_str + 'git_repos/nn_likelihoods/model_paths.yaml'))
model_paths_ccv['fcn_' + method + '_' + str(int(n_data_samples))] = model_path_ccv
yaml.dump(model_paths_ccv, open(folder_str + 'git_repos/nn_likelihoods/model_paths.yaml', "w"))
# -------------------------------------------------------------------------------------------------------------
    
# Making training data
print('Making dataset')
param_grid, boundary_param_grid = generate_param_grid(n_datasets = n_datasets) 
data_grid = generate_data_grid(param_grid, boundary_param_grid)
if len(method_params['boundary_param_names']) > 0:
    param_grid = np.concatenate([param_grid, boundary_param_grid], axis = 1)

print('size of datagrid: ', data_grid.shape)

#Create keras model structure 
model = make_fcn(n_params = param_grid.shape[1])

# Keras callbacks
# Define callbacks
if machine == 'x7':
    ckpt_filename = model_path_x7 + "/model.h5"
    csv_log_filename = model_path_x7 + "/history.csv"
if machine == 'ccv':
    ckpt_filename = model_path_ccv + "/model.h5"
    csv_log_filename = model_path_ccv + "/history.csv"

checkpoint = keras.callbacks.ModelCheckpoint(ckpt_filename, 
                                             monitor = 'val_loss', 
                                             verbose = 1, 
                                             save_best_only = False)
                               
earlystopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                              min_delta = 0, 
                                              verbose = 1, 
                                              patience = 6)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                              factor = 0.1,
                                              patience = 3, 
                                              verbose = 1,
                                              min_delta = 0.0001,
                                              min_lr = 0.0000001)

csv_logger = keras.callbacks.CSVLogger(csv_log_filename)

# Fit model
model.compile(loss = heteroscedastic_loss, 
              optimizer = "adam")

history = model.fit(data_grid, param_grid, 
                    validation_split = .01,
                    batch_size = 32, 
                    epochs = 250, 
                    callbacks = [checkpoint, reduce_lr, earlystopping, csv_logger])

print(history)

if machine == 'x7':
    model.save(model_path_x7 + "/model_final.h5")
if machine == 'ccv':
    model.save(model_path_ccv + "/model_final.h5")