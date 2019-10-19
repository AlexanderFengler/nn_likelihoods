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
machine = 'x7'
method = 'ddm_analytic'
analytic = True
type_of_experiment = 'uniform' # 'uniform', 'perturbation_experiment'
n_parameter_sets = 1000
n_reps = 10
n_data_samples = 3000
out_file_signature = 'v1'

n_experiments = 100
n_datasets_by_experiment = 1

#file_id = sys.argv[1]
#print('argument list: ', str(sys.argv))

# IF WE WANT TO USE A PREVIOUS SET OF PARAMETERS: FOR COMPARISON OF POSTERIORS FOR EXAMPLE
param_origin = 'previous'
param_file_signature  = 'kde_sim_test_ndt_'

if machine == 'x7':
    method_comparison_folder = "/media/data_cifs/afengler/data/kde/ddm/method_comparison/"
if machine == 'ccv':
    method_comparison_folder = '/users/afengler/data/kde/ddm/method_comparison/'

if machine == 'x7':
    stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    output_folder = method_params['output_folder_x7']
if machine == 'ccv':
    stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    output_folder = method_params['output_folder']
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# v, a, w, ndt. angle
def param_grid_perturbation_experiment(n_experiments = n_experiments,
                                       n_datasets_by_experiment = n_datasets_by_experiment,
                                       perturbation_sizes = [[0.05, 0.1, 0.2],
                                                             [0.05, 0.1, 0.2],
                                                             [0.05, 0.1, 0.2],
                                                             [0.05, 0.1, 0.2],
                                                             [0.05, 0.1, 0.2]],
#                                       file_signature = '..',
#                                       file_folder = '..',
                                      method_params = {'0': 0}):
    
    n_perturbation_levels = len(perturbation_sizes[0])
    n_params = len(method_params['param_names']) + len(method_params['boundary_param_names'])
    param_bounds = method_params['param_bounds_sampler'] + method_params['boundary_param_bounds_sampler']
    params_upper_bnd = [bnd[0] for bnd in param_bounds]
    params_lower_bnd = [bnd[1] for bnd in param_bounds]
    total_row_cnt = n_experiments * n_params * (n_perturbation_levels) * n_datasets_by_experiment + n_experiments
    
    
    meta_dat = pd.DataFrame(np.zeros((total_row_cnt, 3)), 
                            columns = ['n_exp', 'param', 'perturbation_level'])
                                            
    param_grid = np.zeros((total_row_cnt, n_params))
    
    cnt = 0
    for i in range(n_experiments):
        param_grid_tmp = np.random.uniform(low = params_lower_bnd, 
                                           high = params_upper_bnd) 
                                           #size = (1, n_params))
        param_grid[cnt, :] = param_grid_tmp
        meta_dat.loc[cnt, :] = [int(i), -1, -1]
        cnt += 1
        for p in range(n_params):
            for l in range(n_perturbation_levels):
                param_grid_perturbed = param_grid_tmp.copy()
                if param_grid_tmp[p] > ((params_upper_bnd[p] - params_lower_bnd[p]) / 2):
                    param_grid_perturbed[p] += perturbation_sizes[p][l]
                else:
                    param_grid_perturbed[p] -= perturbation_sizes[p][l]
                
                param_grid[cnt, :] = param_grid_perturbed
                meta_dat.loc[cnt, :] = [int(i), int(p), int(l)]
                cnt += 1
    
#     pickle.dump(param_grid, open(file_folder + 'param_grid_perturbation_experiment_' + file_signature + '.pickle', 'wb'))
#     meta_dat.to_pickle(file_folder + 'meta_data_perturbation_experiment_' + file_signature)
                
    return (param_grid, meta_dat)

def param_grid_uniform(n = 1000,
                       n_reps = 1,
#                        file_signature = '..',
#                        file_folder = '..',
                       method_params = {'0': 0}):
    
    n_params = len(method_params['param_names']) + len(method_params['boundary_param_names'])
    param_bounds = method_params['param_bounds_sampler'] + method_params['boundary_param_bounds_sampler']
    params_upper_bnd = [bnd[0] for bnd in param_bounds]
    params_lower_bnd = [bnd[1] for bnd in param_bounds]  
    param_grid = np.zeros((n_reps, n,  n_params))
    param_grid[:, :, :] = np.random.uniform(low = params_lower_bnd,
                                   high = params_upper_bnd,
                                   size = (n, n_params))
    
    param_grid = np.reshape(param_grid, (-1, n_params))
    return param_grid 


def generate_data_grid(param_grid, 
                       n_simulations = 10000,
                       method_params = {'0': 0}):
    
    n_parameter_sets = param_grid.shape[0]
    n_params = param_grid.shape[1]
    data_grid = np.zeros((n_parameter_sets, n_simulations, 2))
    n_boundary_params = len(method_params['boundary_param_names'])
    n_process_params = len(method_params["param_names"])
    
    # Split param grid into boundary and 
    for i in range(n_parameter_sets):
        if n_process_params < n_params:
            param_dict_tmp = dict(zip(method_params["param_names"], param_grid[i, :n_process_params]))
            boundary_dict_tmp = dict(zip(method_params["boundary_param_names"], boundary_param_grid[i, n_process_params:]))
        else:
            param_dict_tmp = dict(zip(method_params["param_names"], param_grid[i]))
            boundary_dict_tmp = {}

        rts, choices, _ = method_params["dgp"](**param_dict_tmp, 
                                                   boundary_fun = method_params["boundary"], 
                                                   n_samples = n_simulations,
                                                   delta_t = 0.01, 
                                                   boundary_params = boundary_dict_tmp,
                                                   boundary_multiplicative = method_params['boundary_multiplicative'])

        data_grid[i] = np.concatenate([rts, choices], axis = 1)
        
        if (i % 10) == 0:
            print(str(i) + ' datasets generated')
        
        
        
    return data_grid
# -----------------------------------------------------------------------------

# SIMPLE DATA GRID --------------------------------------------------------------------

# -------------------------------------------------------------------------------------
meta_dat = []

if type_of_experiment == 'perturbation_experiment':
    param_grid, meta_dat = param_grid_perturbation_experiment(n_experiments = 100,
                                                              n_datasets_by_experiment = 1,
                                                              perturbation_sizes = [[0.05, 0.1, 0.2],
                                                                                    [0.05, 0.1, 0.2],
                                                                                    [0.05, 0.1, 0.2],
                                                                                    [0.05, 0.1, 0.2],
                                                                                    [0.05, 0.1, 0.2]],
                                                              file_signature = out_file_signature,
                                                              file_folder = output_folder,
                                                              method_params = method_params,
                                                              save_file = False)
if type_of_experiment == 'uniform':
    param_grid = param_grid_uniform(n = n_parameter_sets,
                                    n_reps = n_reps,
                                    method_params = method_params)

    #                                     file_signature = out_file_signature,
#                                     file_folder = output_folder,

# Generate the corresponding data grid
data_grid = generate_data_grid(param_grid, 
                               n_simulations = n_data_samples,
                               method_params = method_params)


# Dump file
pickle.dump((data_grid, param_grid, meta_dat), open(output_folder + 'base_data_param_recov_unif_reps_' + str(int(n_reps)) \
                                                    + '_n_' + str(int(n_data_samples)) + '.pickle', 'wb'))
