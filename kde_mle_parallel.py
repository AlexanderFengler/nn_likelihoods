# Load packages
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import pickle
import time
import uuid
import scipy as scp
import scipy.stats as scps
from scipy.optimize import differential_evolution
from datetime import datetime
import yaml

# Load my own functions
import keras_to_numpy as ktnp
from kde_training_utilities import kde_load_data # Want to overcome
import cddm_data_simulation as cds
import boundary_functions as bf

# SUPPORT FUNCTIONS ------------
def make_params(param_bounds = []):
    params = np.zeros(len(param_bounds))
    
    for i in range(len(params)):
        params[i] = np.random.uniform(low = param_bounds[i][0], high = param_bounds[i][1])
        
    return params

def get_params_from_meta_data(file_path  = ''):
    # Loading meta data file (,,) (simulator output at this point)
    tmp = pickle.load(open(file_path, 'rb'))[2]
    params = []
    # for loop makes use of common structure of simulator outputs across models
    for key in tmp.keys():
        # delta_t signifies start of simulator parameters that we don't care about for our purposes here
        if key == 'delta_t':
            break
        # variance parameter not used thus far, others added
        if key != 's':
            params.append(key)
    return params 

# Define the likelihood function # NOT USED ATM
# def log_p(params = [0, 1, 0.9], model = [], data = [], ll_min = 1e-29):
#     # Make feature array
#     feature_array = np.zeros((data[0].shape[0], len(params) + 2))
    
#     # Store parameters
#     cnt = 0
#     for i in range(0, len(params), 1):
#         feature_array[:, i] = params[i]
#         cnt += 1
    
#     # Store rts and choices
#     feature_array[:, cnt] = data[0].ravel() # rts
#     feature_array[:, cnt + 1] = data[1].ravel() # choices
    
#     # Get model predictions
#     prediction = np.maximum(model.predict(feature_array), ll_min)
    
#     return(- np.sum(np.log(prediction)))  
# -------------------------------

if __name__ == "__main__":
    
#     Handle some cuda business (if desired to use cuda here..)
    print('Handle cuda business....')   
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    print(device_lib.list_local_devices())
    
    # Initializations -------------
    print('Running intialization....')
    
    # Get configuration from yaml file
    print('Reading config file: ')
    yaml_config_path = os.getcwd() + '/kde_mle_parallel.yaml' # MANUAL INTERVENTION
    with open(yaml_config_path, 'r') as stream:
        config_data = yaml.unsafe_load(stream)

    # Load Model
    print('Loading model: ')
    
    model_path = config_data['model_path']
    ckpt_path = config_data['ckpt_path']
    
    model = keras.models.load_model(model_path)
    model.load_weights(ckpt_path)
    
    # get network architecture for numpy forward pass (used in mle, coming from ktnp imported)
    weights, biases, activations = ktnp.extract_architecture(model)

    print('Setting parameters: ')
    n_runs = config_data['n_runs'] # number of mles to compute in main loop
    n_samples = config_data['n_samples'] # samples by run
    n_workers = config_data['n_workers'] # number of workers to choose for parallel mle
    save_mle_out = config_data['save_mle_out']
    mle_out_path = config_data['mle_out_path']
    param_bounds = config_data['param_bounds']
    param_is_boundary_param = config_data['param_is_boundary_param']
    meta_data_file_path = config_data['meta_data_file_path']
    boundary = eval(config_data['boundary'])
    boundary_multiplicative = config_data['boundary_multiplicative']
    
    # optimizer properties:
    de_optim_popsize = config_data['de_optim_popsize']
    
    # NOTE PARAMETERS: 
    # WEIBULL: [v, a, w, node, shape, scale]
    # LINEAR COLLAPSE: [v, a, w, node, theta]
    # DDM: [v, a, w]
    
    # Get parameter names in correct ordering:
    parameter_names = get_params_from_meta_data(file_path = meta_data_file_path)

    # Make columns for optimizer result table
    p_sim = []
    p_mle = []

    for parameter_name in parameter_names:
        p_sim.append(parameter_name + '_sim')
        p_mle.append(parameter_name + '_mle')

    my_optim_columns = p_sim + p_mle + ['n_samples']

    # Initialize the data frame in which to store optimizer results
    optim_results = pd.DataFrame(np.zeros((n_runs, len(my_optim_columns))), columns = my_optim_columns)
    optim_results.iloc[:, 2 * len(parameter_names)] = n_samples

    # Main loop -------------------------------------------------------------
    for i in range(0, n_runs, 1): 

        # Get start time
        start_time = time.time()

        # Generate set of parameters
        tmp_params = make_params(param_bounds = param_bounds)

        # Define boundary parameters 
        boundary_params = {}
        cnt = 0
        for param in parameter_names:
            if param_is_boundary_param[cnt]:
                boundary_params[parameter] = tmp_params[cnt]
            cnt += 1
            
        # Store in output file
        optim_results.iloc[i, :len(parameter_names)] = tmp_params

        # Print some info on run
        print('Parameters for run ' + str(i) + ': ')
        print(tmp_params)
        
        # Run model simulations: MANUAL INTERVENTION TD: AUTOMATE
        ddm_dat_tmp = cds.ddm_flexbound(v = tmp_params[0],
                                        a = tmp_params[1],
                                        w = tmp_params[2],
                                        s = 1,
                                        delta_t = 0.001,
                                        max_t = 20,
                                        n_samples = n_samples,
                                        boundary_fun = boundary, # function of t (and potentially other parameters) that takes in (t, *args)
                                        boundary_multiplicative = boundary_multiplicative, # CAREFUL: CHECK IF BOUND
                                        boundary_params = boundary_params)

        # Print some info on run
        print('Mean rt for current run: ')
        print(np.mean(ddm_dat_tmp[0]))

        # Run optimizer parallel
        print('Running Optimizer:')
        data_np = np.concatenate([ddm_dat_tmp[0], ddm_dat_tmp[1]], axis = 1)
        out_parallel = differential_evolution(ktnp.log_p, 
                                              bounds = param_bounds,
                                              args = (weights, biases, activations, data_np),
                                              popsize = de_optim_popsize,
                                              disp = True, 
                                              workers = n_workers)

        print('Solution vector of current run: ')
        print(out_parallel.x)

        print('The run took: ')
        elapsed = time.time() - start_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))

        # Store result in output file
        optim_results.iloc[i, len(parameter_names):(2*len(parameter_names))] = out_parallel.x
    # ----------------------------------------------------------
    if save_mle_out:
        # Save optimization results to file
        optim_results.to_csv(mle_out_path + '/mle_results_' + uuid.uuid1().hex + '.csv')

   # NOT USED ATM ----------------------------------------------
#         Run optimizer standard
#         print('running sequential')
#         start_time_sequential = time.time()
#         out = differential_evolution(log_p, 
#                                      bounds = param_bounds, 
#                                      args = (model, ddm_dat_tmp), 
#                                      popsize = 30,
#                                      disp = True)
#         elapsed_sequential = time.time() - start_time_sequential
#         print(time.strftime("%H:%M:%S", time.gmtime(elapsed_sequential)))

        # Print some info
#         print('Solution vector of current run: ')
#         print(out.x)