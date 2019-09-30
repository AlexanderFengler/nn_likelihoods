import tensorflow as tf
from tensorflow import keras
import numpy as np
from cdwiener import array_fptd
import os
import pandas as pd
import time
from datetime import datetime
import pickle
import yaml
import keras_to_numpy as ktnp

from kde_training_utilities import kde_load_data
from kde_training_utilities import kde_make_train_test_split

# CHOOSE ---------
method = "angle_ndt" # ddm, linear_collapse, ornstein, full, lba
machine = 'x7'
# ----------------

# INITIALIZATIONS ----------------------------------------------------------------
stats = pickle.load(open("kde_stats.pickle", "rb"))[method]
dnn_params = yaml.load(open("hyperparameters.yaml"))

if machine == 'x7':
    data_folder = stats["data_folder_x7"]
    model_path = stats["model_folder_x7"]
else:
    data_folder = stats["data_folder"]
    model_path = stats["model_folder"]
    
model_path += dnn_params["model_type"] + "_{}_".format(method) + datetime.now().strftime('%m_%d_%y_%H_%M_%S') + "/"

print('if it does not exist, make model path')

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
# Copy hyperparameter setup into model path
if machine == 'x7':
    os.system("cp {} {}".format("/media/data_cifs/afengler/git_repos/nn_likelihoods/hyperparameters.yaml", model_path))
else:
    os.system("cp {} {}".format("/users/afengler/git_repos/nn_likelihoods/hyperparameters.yaml", model_path))
    
# set up gpu to use
if machine == 'x7':
    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = dnn_params['gpu_x7'] 

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Load the training data
print('loading data.... ')

X, y, X_val, y_val = kde_load_data(folder = data_folder, 
                                   return_log = True, # Dont take log if you want to train on actual likelihoods
                                   prelog_cutoff = 1e-7 # cut out data with likelihood lower than 1e-7
                                  )

X = np.array(X)
X_val = np.array(X_val)
# --------------------------------------------------------------------------------

# MAKE MODEL ---------------------------------------------------------------------
print('Setting up keras model')

input_shape = X.shape[1]
model = keras.Sequential()

for i in range(len(dnn_params['hidden_layers'])):
    if i == 0:
        model.add(keras.layers.Dense(units = dnn_params["hidden_layers"][i], 
                                     activation = dnn_params["hidden_activations"][i], 
                                     input_dim = input_shape))
    else:
        model.add(keras.layers.Dense(units = dnn_params["hidden_layers"][i],
                                     activation = dnn_params["hidden_activations"][i]))
        
# Write model specification to yaml file        
spec = model.to_yaml()
open(model_path + "model_spec.yaml", "w").write(spec)


print('STRUCTURE OF GENERATED MODEL: ....')
print(model.summary())

if dnn_params['loss'] == 'huber':
    model.compile(loss = tf.losses.huber_loss, 
                  optimizer = "adam", 
                  metrics = ["mse"])

if dnn_params['loss'] == 'mse':
    model.compile(loss = 'mse', 
                  optimizer = "adam", 
                  metrics = ["mse"])
# ---------------------------------------------------------------------------

# FIT MODEL -----------------------------------------------------------------
print('Starting to fit model.....')

# Define callbacks
ckpt_filename = model_path + "model.h5"

checkpoint = keras.callbacks.ModelCheckpoint(ckpt_filename, 
                                             monitor = 'val_loss', 
                                             verbose = 1, 
                                             save_best_only = False)
                               
earlystopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                              min_delta = 0, 
                                              verbose = 1, 
                                              patience = 2)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                              factor = 0.1,
                                              patience = 1, 
                                              verbose = 1,
                                              min_delta = 0.0001,
                                              min_lr = 0.0000001)

history = model.fit(X, y, 
                    validation_data = (X_val, y_val), 
                    epochs = dnn_params["n_epochs"],
                    batch_size = dnn_params["batch_size"], 
                    callbacks = [checkpoint, reduce_lr, earlystopping], 
                    verbose = 2)
# ---------------------------------------------------------------------------

# SAVING --------------------------------------------------------------------
print('Saving model and relevant data...')
# Log of training output
pd.DataFrame(history.history).to_csv(model_path + "training_history.csv")

# Save Model
model.save(model_path + "model_final.h5")

# Extract model architecture as numpy arrays and save in model path
__, ___, ____, = ktnp.extract_architecture(model, save = True, save_path = model_path)

# Update model paths in model_path.yaml
model_paths = yaml.load(open("model_paths.yaml"))
model_paths[method] = model_path
yaml.dump(model_paths, open("model_paths.yaml", "w"))
# ----------------------------------------------------------------------------

# UNUSED --------------------------

# def extract_info(model):
#     biases = []
#     activations = []
#     weights = []
#     for layer in model.layers:
#         if layer.name == "input_1":
#             continue
#         weights.append(layer.get_weights()[0])
#         biases.append(layer.get_weights()[1])
#         activations.append(layer.get_config()["activation"])
#     return weights, biases, activations

# weights, biases, activations = extract_info(model)

# pickle.dump(weights, open(model_path + "weights.pickle" ,"wb"))
# pickle.dump(biases, open(model_path + "biases.pickle" ,"wb"))
# pickle.dump(activations, open(model_path + "activations.pickle" ,"wb"))

# kde_make_train_test_split(folder = data_folder + "/",
#                           p_train = 0.99)

# Seems unnecessary
# if not params["log"]:
#     y = np.exp(y)
