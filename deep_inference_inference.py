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
import re

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from cdwiener import batch_fptd
import clba

# INITIALIZATIONS -------------------------------------------------------------
# print(device_lib.list_local_devices())

#machine = 'x7'
method = 'ddm_ndt'
fcn_custom_objects = {"heteroscedastic_loss": tf.losses.huber_loss}

def load_data_perturbation_experiment(file_ = '..'):
    tmp = pickle.load(open(file_, 'rb'))
    data_grid = tmp[1]
    param_grid = tmp[0]
    return (param_grid, data_grid)
    

def run_inference(file_list = ['.', '.'],
                  machine = 'x7',
                  method = 'ddm_ndt'):
    
    # Setup
    if machine == 'x7':
        dnn_params = yaml.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/hyperparameters.yaml"))
        stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        method_params = stats[method]
        os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = dnn_params['gpu_x7'] 
        
        print(' GPU I AM ASKING FOR: ', dnn_params['gpu_x7'])
        
        with open("model_paths_x7.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)['fcn_' + method]

    if machine == 'ccv':
        stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        method_params = stats[method]
        with open("model_paths.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)['fcn_' + method]
    
    # Load Model
    fcn = keras.models.load_model(network_path + 'model_final.h5', custom_objects = fcn_custom_objects)

    # Run inference and store inference files for all files in file_list
    for file_ in file_list:
        param_grid, data_grid = load_data_perturbation_experiment(file_ = file_)
        fcn_results = fcn.predict(data_grid)
        
        pattern_span = re.search('base_data', file_).span()
        tmp_file_name = file_[:pattern_span[0]] + 'deep_inference' + file_[pattern_span[1]:]
        
        pickle.dump((param_grid, data_grid, fcn_results), 
                    open(tmp_file_name, "wb"))
        print(file_)