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
    return keras.backend.sum(precision * (true - point) ** 2 + keras.backend.log(var), - 1)


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

data_folder = "/users/afengler/data/kde/full_ddm/train_test_data_20000/"
files = os.listdir(data_folder)
files = [f for f in files if re.match("data_.*", f)]

timestamp = datetime.now().strftime('%m_%d_%y_%H_%M_%S')
# save model configuration and hyperparams in folder

# INITIALIZATIONS -------------------------------------------------------------
machine = 'ccv'
method = 'ddm_ndt'
analytic = True
file_signature = '_start_true_'
n_data_samples = 2500
n_cpus = 'all'

#stats = pickle.load(open("kde_stats.pickle", "rb"))
#method_params = stats[method]

if machine == 'x7':
    stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    #output_folder = method_params['output_folder_x7']
    model_folder = method_paramsp['model_folder_x7']
if machine == 'ccv':
    stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    #output_folder = method_params['output_folder']
    model_folder = method_params['model_folder']
        
print(stats)
print(method_params)

# Specify final model path
model_path = model_folder + "deep_inference_" + timestamp

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create keras model structure 
model = make_model()

# Keras callbacks
earlystopper = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 2, verbose = 1)
checkpoint = keras.callbacks.ModelCheckpoint(model_path + '/model.h5', monitor = "val_loss", verbose = 1, save_best_only = False)
csv_logger = CSVLogger('history.csv')
# Fit model

model.compile(loss = heteroscedastic_loss, optimizer = "adam")
history = model.fit(X, y, 
                    validation_split = .01, 
                    epochs = 10, 
                    batch_size = 32, 
                    callbacks = [checkpoint, earlystopper, csv_logger])

print(history)

model.save(model_path + "/model_final.h5")