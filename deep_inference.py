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
        param_upper_bnd.append(method_params['param_bounds_sampler'][p][1])
        param_lower_bnd.append(method_params['param_bounds_sampler'][p][0])

    if len(method_params['boundary_param_names']) > 0:
        for p in range(len(method_params['boundary_param_names'])):
            boundary_param_upper_bnd.append(method_params['boundary_param_bounds_sampler'][p][1])
            boundary_param_lower_bnd.append(method_params['boundary_param_bounds_sampler'][p][0])                                    

    param_grid = np.random.uniform(low = param_lower_bnd, 
                                   high = param_upper_bnd, 
                                   size = (n_datasets, len(method_params['param_names'])))

    if len(method_params['boundary_param_names']) > 0:
        boundary_param_grid = np.random.uniform(low = boundary_param_lower_bnd,
                                                high = boundary_param_upper_bnd,
                                                size = (n_datasets, len(method_params['boundary_param_bounds_sampler'])))
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
method = 'angle_ndt'
analytic = True
file_signature = '_start_true_'
n_data_samples = 2500
n_datasets = 500000
n_cpus = 'all'


dnn_params = yaml.load(open("hyperparameters.yaml"))
# set up gpu to use
if machine == 'x7':
    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = dnn_params['gpu_x7'] 

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#stats = pickle.load(open("kde_stats.pickle", "rb"))
#method_params = stats[method]

if machine == 'x7':
    stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    #output_folder = method_params['output_folder_x7']
    model_folder = method_params['model_folder_x7']
if machine == 'ccv':
    stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    #output_folder = method_params['output_folder']
    model_folder = method_params['model_folder']
        
#print(stats)
#print(method_params)
# Specify final model path
model_path = model_folder + "deep_inference_" + timestamp

if not os.path.exists(model_path):
    os.makedirs(model_path)

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
ckpt_filename = model_path + "/model.h5"

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

#earlystopper = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, verbose = 1)
#checkpoint = keras.callbacks.ModelCheckpoint(model_path + '/model.h5', monitor = "val_loss", verbose = 1, save_best_only = False)

csv_logger = keras.callbacks.CSVLogger(model_path + '/history.csv')

# Fit model
model.compile(loss = heteroscedastic_loss, optimizer = "adam")
history = model.fit(data_grid, param_grid, 
                    validation_split = .01,
                    batch_size = 32, 
                    epochs = 250, 
                    callbacks = [checkpoint, reduce_lr, earlystopping, csv_logger])

print(history)
model.save(model_path + "/model_final.h5")