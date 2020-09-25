# MAIN SCRIPT FOR DATASET GENERATION

# IMPORTS ------------------------------------------------------------------------
import numpy as np
import yaml
import pandas as pd
from itertools import product
import pickle
import uuid
import os
import sys
from datetime import datetime
from scipy.stats import truncnorm

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
import basic_simulator as bs
import kde_info

#from tqdm as tqdm
#from cdwiener import batch_fptd
#import clba

# Parallelization
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Pool
import psutil
import argparse
from functools import partial

# --------------------------------------------------------------------------------

# Data generator class that generates datasets for us ----------------------------
class data_generator():
    def __init__(self,
                 config = None):
    # INIT -----------------------------------------
        if config == None:
            print()
            return
        else:
            self.config = config
            self._build_simulator()
            self._get_ncpus()
            
    def _get_training_data_theta(self, theta):
        out = self.get_simulations(theta)
    
    def _filter_simulations_fast(self,
                                 simulations = None,
                                 filters = {'mode': 20, # != (checking if mode is max_rt)
                                           'choice_cnt': 0, # > (checking that each choice receive at least 10 samples in simulator)
                                           'mean_rt': 15, # < (checking that mean_rt is smaller than specified value
                                           'std': 0, # > (checking that std is positive for each choice)
                                           'mode_cnt_rel': 0.5  # < (checking that mode does not receive more than a proportion of samples for each choice)
                                           }
                                ):

        max_t = simulations[2]['max_t']
        tmp_max_rt_ = simulations[0].max().round(2)

        keep = 1
        for choice_tmp in simulations[2]['possible_choices']:
            tmp_rts = simulations[0][simulations[1] == choice_tmp]
            tmp_n_c = len(tmp_rts)

            if n_c > 0:
                mode_, mode_cnt_ = mode(tmp_rts)
                std_ = np.std(tmp_rts)
                mean_ = np.mean(tmp_rts)
            else:
                mode_ = -1
                mode_cnt_ = 0
                mean_ = -1
                std_ = -1

            mode_cnt_rel_ = mode_cnt_ / tmp_n_c
            choice_prop_  = tmp_n_c / simulations[2]['n_samples']  

            keep = keep & \
                   (mode_ != filters['mode']) & \
                   (choice_cnt > filters['choice_cnt']) & \
                   (mean_ < filters['mean_rt']) & \
                   (std_ > filters['std']) & \
                   (mode_cnt_rel < filters['mode_cnt_rel']) & \
                   (tmp_n_c > filters['choice_cnt'])
        return keep
             
    def _make_kde_data(self,
                       simulations = None, 
                       theta = None
                       n_kde = 800, 
                       n_unif_up = 100, 
                       n_unif_down = 100):
        
        out = np.zeros((n_kde + n_unif_up + n_unif_down, 
                        3 + len(theta)))
        out[:len(theta), :] = np.tile(theta, (n_kde + n_unif_up + n_unif_down, 1) )
        
        tmp_kde = kde_class.logkde((simulations[0],
                                    simulations[1], 
                                    simulations[2]))

        # Get kde part
        samples_kde = tmp_kde.kde_sample(n_samples = n_kde)
        likelihoods_kde = tmp_kde.kde_eval(data = samples_kde).ravel()

        out[:n_kde, 0] = samples_kde[0].ravel()
        out[:n_kde, 1] = samples_kde[1].ravel()
        out[:n_kde, 2] = likelihoods_kde

        # Get positive uniform part:
        choice_tmp = np.random.choice(metadata['possible_choices'],
                                      size = n_unif_up)

        if metadata['max_t'] < 100:
            rt_tmp = np.random.uniform(low = 0.0001,
                                       high = metadata['max_t'],
                                       size = n_unif_up)
        else: 
            rt_tmp = np.random.uniform(low = 0.0001, 
                                       high = 100,
                                       size = n_unif_up)

        likelihoods_unif = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp)).ravel()


        out[n_kde:(n_kde + n_unif_up), 0] = rt_tmp
        out[n_kde:(n_kde + n_unif_up), 1] = choice_tmp
        out[n_kde:(n_kde + n_unif_up), 2] = likelihoods_unif


        # Get negative uniform part:
        choice_tmp = np.random.choice(metadata['possible_choices'],
                                      size = n_unif_down)

        rt_tmp = np.random.uniform(low = - 1.0,
                                   high = 0.0001,
                                   size = n_unif_down)

        out[(n_kde + n_unif_up):, -3] = rt_tmp
        out[(n_kde + n_unif_up):, -2] = choice_tmp
        out[(n_kde + n_unif_up):, -1] = -66.77497

        if idx % 10 == 0:
            print(idx)

        return out.astype(np.float)
    
    def _get_processed_data_for_theta(self,
                                      theta)
    
        keep = 0
        while not keep:
            simulations = self.get_simulations()
            keep = self.filter_simulations_fast(simulations,
                                                filters = {'mode': 20, # != (checking if mode is max_rt)
                                                           'choice_cnt': 0, # > (checking that each choice receive at least 10 samples in simulator)
                                                           'mean_rt': 15, # < (checking that mean_rt is smaller than specified value
                                                           'std': 0, # > (checking that std is positive for each choice)
                                                          'mode_cnt_rel': 0.5  # < (checking that mode does not receive more than a proportion of samples for each choice)
                                                          }
                                               )
        
        data = self._make_kde_data(simulations = simulations,
                                   theta = theta,
                                   n_kde = 800,
                                   n_unif_up = 100,
                                   n_unif_down = 100)
        
        return data
             
    def _get_ncpus(self):
        
        # Sepfic
        if self.config['n_cpus'] == 'all':
            n_cpus = psutil.cpu_count(logical = False)
        else:
            n_cpus = self.config['n_cpus']
        
        self.config['n_cpus'] = n_cpus
        
    def _build_simulator(self):
        self.simulator = partial(bs.simulator, 
                                 n_samples = self.config['n_samples'],
                                 max_t = self.config['max_t'],
                                 bin_dim = self.config['nbins'],
                                 delta_t = self.config['delta_t'])
                                 
    def get_simulations(self, theta = None):
        out = self.simulator(theta, 
                             self.config['method'])
        # TODO: Add 
        if self.config['nbins'] is not None:
            return np.concatenate([out[0], out[1]], axis = 1)
        else:
            return out
        
    def generate_full_data_uniform(self, 
                                   save = False):
        
        # Make parameters
        theta_list = [np.float32((np.random.uniform(low = self.config['param_bounds'][0], 
                                                    high = self.config['param_bounds'][1])) for i in range(self.config['nparamsets']))]
        
        # Get simulations
        with Pool(processes = self.config['n_cpus']) as pool:
            data_grid = np.array(pool.starmap(self._get_processed_data_for_theta, theta_list))
         
        
        # Save to correct destination
        if save:
            
            # -----
            training_data_folder = self.config['method_folder'] + \
                                  'training_data_binned_' + \
                                  str(int(self.config['binned'])) + \
                                  '_nbins_' + str(self.config['nbins']) + \
                                  '_n_' + str(self.config['nsamples'])

            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)

            full_file_name = training_data_folder + '/' + \
                            self.config['method'] + \
                            '_nchoices_' + str(self.config['nchoices']) + \
                            '_train_data_binned_' + \
                            str(int(self.config['binned'])) + \
                            '_nbins_' + str(self.config['nbins']) + \
                            '_n_' + str(self.config['nsamples']) + \
                            '_' + self.file_id + '.pickle'
            
            print('Writing to file: ', full_file_name)
            
            pickle.dump((np.float32(np.stack(theta_list)), 
                         np.float32(np.expand_dims(data_grid, axis = 0)), 
                         self.config['meta']), 
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            
            return 'Dataset completed'
        
        # Or else return the data
        else:
            return np.float32(np.stack(theta_list)), np.float32(np.expand_dims(data_grid, axis = 0)) 
                                                             
    def generate_data_uniform(self, save = False):
        
        # Make parameters
        theta_list = [np.float32((np.random.uniform(low = self.config['param_bounds'][0], 
                                                    high = self.config['param_bounds'][1])) for i in range(self.config['nparamsets']))]
        
        # Get simulations
        with Pool(processes = self.config['n_cpus']) as pool:
            data_grid = np.array(pool.starmap(self.get_simulations, theta_list))
         
        
        # Save to correct destination
        if save:
            
            # -----
            if self.config['mode'] == 'test':
                training_data_folder = self.config['method_folder'] + \
                                       'parameter_recovery_data_binned_' + \
                                       str(int(self.config['binned'])) + \
                                       '_nbins_' + str(self.config['nbins']) + \
                                       '_n_' + str(self.config['nsamples'])
                
                if not os.path.exists(training_data_folder):
                    os.makedirs(training_data_folder)

                full_file_name = training_data_folder + '/' + \
                                self.config['method'] + \
                                '_nchoices_' + str(self.config['nchoices']) + \
                                '_parameter_recovery_binned_' + \
                                str(int(self.config['binned'])) + \
                                '_nbins_' + str(self.config['nbins']) + \
                                '_nreps_' + str(self.config['nreps']) + \
                                '_n_' + str(self.config['nsamples']) + \
                                '.pickle'
            
            else:
                training_data_folder = self.config['method_folder'] + \
                                      'training_data_binned_' + \
                                      str(int(self.config['binned'])) + \
                                      '_nbins_' + str(self.config['nbins']) + \
                                      '_n_' + str(self.config['nsamples'])
                
                if not os.path.exists(training_data_folder):
                    os.makedirs(training_data_folder)

                full_file_name = training_data_folder + '/' + \
                                self.config['method'] + \
                                '_nchoices_' + str(self.config['nchoices']) + \
                                '_train_data_binned_' + \
                                str(int(self.config['binned'])) + \
                                '_nbins_' + str(self.config['nbins']) + \
                                '_n_' + str(self.config['nsamples']) + \
                                '_' + self.file_id + '.pickle'
            
            print('Writing to file: ', full_file_name)
            
            pickle.dump((np.float32(np.stack(theta_list)), 
                         np.float32(np.expand_dims(data_grid, axis = 0)), 
                         self.config['meta']), 
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            
            return 'Dataset completed'
        
        # Or else return the data
        else:
            return np.float32(np.stack(theta_list)), np.float32(np.expand_dims(data_grid, axis = 0)) 
            
    def generate_data_hierarchical(self, save = False):
        
        subject_param_grid, global_stds, global_means = self._make_param_grid_hierarchical()
        subject_param_grid_adj_sim = np.reshape(subject_param_grid, (-1, self.config['nparams'])).tolist()
        subject_param_grid_adj_sim = tuple([(np.array(i),) for i in subject_param_grid_adj_sim])
        
        with Pool(processes = self.config['n_cpus']) as pool:
            data_grid = np.array(pool.starmap(self.get_simulations, subject_param_grid_adj_sim))
            
        if save:
            training_data_folder = self.config['method_folder'] + 'parameter_recovery_hierarchical_data_binned_' + str(int(self.config['binned'])) + \
                                   '_nbins_' + str(self.config['nbins']) + \
                                   '_n_' + str(self.config['nsamples'])
            
            full_file_name = training_data_folder + '/' + \
                             self.config['method'] + \
                             '_nchoices_' + str(self.config['nchoices']) + \
                             '_parameter_recovery_hierarchical_' + \
                             'binned_' + str(int(self.config['binned'])) + \
                             '_nbins_' + str(self.config['nbins']) + \
                             '_nreps_' + str(self.config['nreps']) + \
                             '_n_' + str(self.config['nsamples']) + \
                             '_nsubj_' + str(self.config['nsubjects']) + \
                             '.pickle'
            
            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)
            
            print('saving dataset as ', full_file_name)
            
            pickle.dump(([subject_param_grid, global_stds, global_means], 
                          np.expand_dims(data_grid, axis = 0),
                          self.config['meta']), 
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            
            return 'Dataset completed'
        else:
            return ([subject_param_grid, global_stds, global_means], data_grid, meta)
  
    def _make_param_grid_hierarchical(self):
        # Initialize global parameters

        params_ranges_half = (np.array(self.config['param_bounds'][1]) - np.array(self.config['param_bounds'][0])) / 2
        
        # Sample global parameters from cushioned parameter space
        global_stds = np.random.uniform(low = 0.001,
                                        high = params_ranges_half / 10,
                                        size = (self.config['nparamsets'], self.config['nparams']))
        global_means = np.random.uniform(low = self.config['param_bounds'][0] + (params_ranges_half / 5),
                                         high = self.config['param_bounds'][1] - (params_ranges_half / 5),
                                         size = (self.config['nparamsets'], self.config['nparams']))

        # Initialize local parameters (by condition)
        subject_param_grid = np.float32(np.zeros((self.config['nparamsets'], self.config['nsubjects'], self.config['nparams'])))
        
        # Sample by subject parameters from global setup (truncate to never go out of allowed parameter space)
        for n in range(self.config['nparamsets']):
            for i in range(self.config['nsubjects']):
                a, b = (self.config['param_bounds'][0] - global_means[n]) / global_stds[n], (self.config['param_bounds'][1] - global_means[n]) / global_stds[n]
                subject_param_grid[n, i, :] = np.float32(global_means[n] + truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[n])

        return subject_param_grid, global_stds, global_means
   # ----------------------------------------------------
 
# -------------------------------------------------------------------------------------

# RUN 
if __name__ == "__main__":
    # Make command line interface
    CLI = argparse.ArgumentParser()
    
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--dgplist", 
                     nargs = "*",
                     type = str,
                     default = ['ddm', 'ornstein', 'angle', 'weibull', 'full_ddm'])
    CLI.add_argument("--datatype",
                     type = str,
                     default = 'uniform')
    CLI.add_argument("--mode",
                     type = str,
                     default = 'test') # 'parameter_recovery, 'perturbation_experiment', 'r_sim', 'r_dgp', 'cnn_train', 'parameter_recovery_hierarchical'
    CLI.add_argument("--nsubjects",
                    type = int,
                    default = 5)
    CLI.add_argument("--nreps",
                     type = int,
                     default = 1)
    CLI.add_argument("--nbins",
                     type = int,
                     default = None)
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 20000)
    CLI.add_argument("--nchoices",
                     type = int,
                     default = 2)
#     CLI.add_argument("--mode",
#                      type = str,
#                      default = 'mlp') # train, test, cnn
    CLI.add_argument("--nsimbnds",
                     nargs = '*',
                     type = int,
                     default = [100, 100000])
    CLI.add_argument("--nparamsets", 
                     type = int,
                     default = 10000)
    CLI.add_argument("--fileid",
                     type = str,
                     default = 'TEST')
    CLI.add_argument("--save",
                     type = bool,
                     default = 0)
    CLI.add_argument("--maxt",
                     type = float,
                     default = 10.0)
    CLI.add_argument("--deltat",
                     type = float,
                     default = 0.001)
    CLI.add_argument("--pickleprotocol",
                     type = int,
                     default = 4)
    
    args = CLI.parse_args()
    print('Arguments passed: ')
    print(args)
    
    machine = args.machine
    
    # SETTING UP CONFIG --------------------------------------------------------------------------------
    config = {}
    config['n_cpus'] = 'all'
    
    # Update config with specifics of run
    if args.datatype == 'r_dgp':
        config['method'] = args.dgplist
    else:
        config['method'] = args.dgplist[0]
        
    config['mode'] = args.mode
    config['file_id'] = args.fileid
    config['nsamples'] = args.nsamples
    if args.nbins is not None:
        config['binned'] = 1
    else:
        config['binned'] = 0
        
    if args.nbins == 0:
        config['nbins'] = None
    else:
        config['nbins'] = args.nbins
        
    config['datatype'] = args.datatype
    config['nchoices'] = args.nchoices
    config['nparamsets'] = args.nparamsets
    config['nreps'] = args.nreps
    config['pickleprotocol'] = args.pickleprotocol
    config['nsimbnds'] = args.nsimbnds
    config['nsubjects'] = args.nsubjects
    config['n_samples'] = args.nsamples
    config['max_t'] = args.maxt
    config['delta_t'] = args.deltat
    
    # Make parameter bounds
    if args.mode == 'train' and config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_cnn'] + kde_info.temp[config['method']]['boundary_param_bounds_cnn']
    elif args.mode == 'train' and not config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_network'] + kde_info.temp[config['method']]['boundary_param_bounds_network']
    elif args.mode == 'test' and config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_sampler'] + kde_info.temp[config['method']]['boundary_param_bounds_sampler']
    elif args.mode == 'test' and not config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_sampler'] + kde_info.temp[config['method']]['boundary_param_bounds_sampler']

    config['param_bounds'] = np.array([[i[0] for i in bounds_tmp], [i[1] for i in bounds_tmp]])
    config['nparams'] = config['param_bounds'][0].shape[0]
    
    config['meta'] = kde_info.temp[config['method']]['dgp_hyperparameters']
    
    # Add some machine dependent folder structure
    if args.machine == 'x7':
        config['method_comparison_folder'] = kde_info.temp[config['method']]['output_folder_x7']
        config['method_folder'] = kde_info.temp[config['method']]['method_folder_x7']

    if args.machine == 'ccv':
        config['method_comparison_folder'] = kde_info.temp[config['method']]['output_folder']
        config['method_folder'] = kde_info.temp[config['method']]['method_folder']

    if args.machine == 'home':
        config['method_comparison_folder'] = kde_info.temp[config['method']]['output_folder_home']
        config['method_folder'] = kde_info.temp[config['method']]['method_folder_home']

    if args.machine == 'other': # This doesn't use any extra 
        if not os.path.exists('data_storage'):
            os.makedirs('data_storage')

        print('generated new folder: data_storage. Please update git_ignore if this is not supposed to be committed to repo')

        config['method_comparison_folder']  = 'data_storage/'
        config['method_folder'] = 'data_storage/' + config['method'] + '_'
    # -------------------------------------------------------------------------------------
    
    # GET DATASETS ------------------------------------------------------------------------
    # Get data for the type of dataset we want
    start_t = datetime.now()
    
    dg = data_generator(config = config)
    
    if args.datatype == 'parameter_recovery' or args.datatype == 'training':
        dg.generate_data_uniform(save = args.save)
        
    if args.datatype == 'parameter_recovery_hierarchical':
        dg.generate_data_hierarchical(save = args.save)
        
    finish_t = datetime.now()
    print('Time elapsed: ', finish_t - start_t)
    print('Finished')
    # -------------------------------------------------------------------------------------
    
# UNUSED ------------------------------    
    
#     def generate_data_grid_parallel(self,
#                                     param_grid = []):
        
#         args_list = self.make_args_starmap_ready(param_grid = param_grid)
        
#         if self.config['n_cpus'] == 'all':
#             n_cpus = psutil.cpu_count(logical = False)
#         else:
#             n_cpus = self.config['n_cpus']

#         # Run Data generation
#         with Pool(processes = n_cpus) as pool:
#             data_grid = np.array(pool.starmap(self.data_generator, args_list))

#         return data_grid   
  
        
  
#     def clean_up_parameters(self):
        
#         if self.config['mode'] == 'test':
#             param_bounds = self.method_params['param_bounds_sampler'] + self.method_params['boundary_param_bounds_sampler']
#         if self.config['mode'] == 'mlp':
#             param_bounds = self.method_params['param_bounds_network'] + self.method_params['boundary_param_bounds_network']
#         if self.config['mode'] == 'cnn':
#             param_bounds = self.method_params['param_bounds_cnn'] + self.method_params['boundary_param_bounds_cnn']
        
#         # Epsilon correction of boundaries (to make sure for parameter recovery we don't generate straight at the bound)
        
#         eps = 0
#         if self.config['datatype'] == 'parameter_recovery' and self.config['mode'] != 'test':
#             # TD make eps parameter
#             eps = 0.05
            
#         print('epsilon correction', eps)

#         # If model is lba, lca, race we need to expand parameter boundaries to account for
#         # parameters that depend on the number of choices
#         if self.method == 'lba' or self.method == 'lca' or self.method == 'race_model':
#             param_depends_on_n = self.method_params['param_depends_on_n_choice']
#             param_bounds_tmp = []
            
#             n_process_params = len(self.method_params['param_names'])
            
#             p_cnt = 0
#             for i in range(n_process_params):
#                 if self.method_params['param_depends_on_n_choice'][i]:
#                     for c in range(self.config['nchoices']):
#                         param_bounds_tmp.append(param_bounds[i])
#                         p_cnt += 1
#                 else:
#                     param_bounds_tmp.append(param_bounds[i])
#                     p_cnt += 1
            
#             self.method_params['n_process_parameters'] = p_cnt
            
#             param_bounds_tmp += param_bounds[n_process_params:]
#             params_upper_bnd = [bnd[1] - eps for bnd in param_bounds_tmp]
#             params_lower_bnd = [bnd[0] + eps for bnd in param_bounds_tmp]
                
#             #print(params_lower_bnd)
            
            
#         # If our model is not lba, race, lca we use simple procedure 
#         else:
#             params_upper_bnd = [bnd[1] - eps for bnd in param_bounds]
#             params_lower_bnd = [bnd[0] + eps for bnd in param_bounds]
            
#         return params_upper_bnd, params_lower_bnd
                       

#     def make_dataset_perturbation_experiment(self,
#                                              save = True):
        
#         param_grid, meta_dat = self.make_param_grid_perturbation_experiment()
             
#         if self.config['binned']:           
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['n_experiments'],
#                                   param_grid.shape[1], 
#                                   self.config['nbins'],
#                                   self.config['nchoices']))         
        
#         else:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['n_experiments'],
#                                   param_grid.shape[1], 
#                                   self.config['nsamples'],
#                                   2)) 

#         for experiment in range(self.config['n_experiments']):
#             for rep in range(self.config['nreps']):
#                 data_grid[rep, experiment] = self.generate_data_grid_parallel(param_grid = param_grid[experiment])
#                 print(experiment, ' experiment data finished')
        
#         if save == True:
#             print('saving dataset')
#             pickle.dump((param_grid, data_grid, meta_dat), open(self.method_comparison_folder + \
#                                                                 'base_data_perturbation_experiment_nexp_' + \
#                                                                 str(self.config['n_experiments']) + \
#                                                                 '_nreps_' + str(self.config['nreps']) + \
#                                                                 '_n_' + str(self.config['nsamples']) + \
#                                                                 '_' + self.config['file_id'] + '.pickle', 'wb'))
                        
#             return 'Dataset completed'
#         else:
#             return param_grid, data_grid, meta_dat
    
    
# def make_dataset_parameter_recovery(self,
#                                         save = True):
        
#         param_grid = self.make_param_grid_uniform()
#         self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'])]
        
#         if self.config['binned']:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nbins'],
#                                   self.config['nchoices']))
#         else:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nsamples'],
#                                   2))
        
#         for rep in range(self.config['nreps']):
#             data_grid[rep] = np.array(self.generate_data_grid_parallel(param_grid = param_grid))
#             print(rep, ' repetition finished') 
        
        
#         if save:
#             training_data_folder = self.method_folder + 'parameter_recovery_data_binned_' + str(int(self.config['binned'])) + \
#                                    '_nbins_' + str(self.config['nbins']) + \
#                                    '_n_' + str(self.config['nsamples'])
#             if not os.path.exists(training_data_folder):
#                 os.makedirs(training_data_folder)
                
#             full_file_name = training_data_folder + '/' + \
#                             self.method + \
#                             '_nchoices_' + str(self.config['nchoices']) + \
#                             '_parameter_recovery_' + \
#                             'binned_' + str(int(self.config['binned'])) + \
#                             '_nbins_' + str(self.config['nbins']) + \
#                             '_nreps_' + str(self.config['nreps']) + \
#                             '_n_' + str(self.config['nsamples']) + \
#                             '.pickle'
            
#             print(full_file_name)
            
#             meta = self.dgp_hyperparameters.copy()
#             if 'boundary' in meta.keys():
#                 del meta['boundary']

#             pickle.dump((param_grid, data_grid, meta), 
#                         open(full_file_name, 'wb'), 
#                         protocol = self.config['pickleprotocol'])
            
#             return 'Dataset completed'
#         else:
#             return param_grid, data_grid

#     def make_dataset_train_network_unif(self,
#                                         save = True):

#         param_grid = self.make_param_grid_uniform()
#         self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'])]

#         if self.config['binned']:
#             data_grid = np.zeros((self.config['nparamsets'],
#                                   self.config['nbins'],
#                                   self.config['nchoices']))
#         else:
#             data_grid = np.zeros((self.config['nparamsets'],
#                                   self.config['nsamples'],
#                                   2))
#         data_grid = np.array(self.generate_data_grid_parallel(param_grid = param_grid))

#         if save:
#             training_data_folder = self.method_folder + 'training_data_binned_' + str(int(self.config['binned'])) + \
#                                    '_nbins_' + str(self.config['nbins']) + \
#                                    '_n_' + str(self.config['nsamples'])
#             if not os.path.exists(training_data_folder):
#                 os.makedirs(training_data_folder)
                
#             full_file_name = training_data_folder + '/' + \
#                             self.method + \
#                             '_nchoices_' + str(self.config['nchoices']) + \
#                             '_train_data_' + \
#                             'binned_' + str(int(self.config['binned'])) + \
#                             '_nbins_' + str(self.config['nbins']) + \
#                             '_n_' + str(self.config['nsamples']) + \
#                             '_' + self.file_id + '.pickle'
            
#             print(full_file_name)
            
#             meta = self.dgp_hyperparameters.copy()
#             if 'boundary' in meta.keys():
#                 del meta['boundary']

#             pickle.dump((param_grid, data_grid, meta), 
#                         open(full_file_name, 'wb'), 
#                         protocol = self.config['pickleprotocol'])
#             return 'Dataset completed'
#         else:
#             return param_grid, data_grid

#     def make_param_grid_hierarchical(self,
#                                      ):

#         # Initializations
#         params_upper_bnd, params_lower_bnd = self.clean_up_parameters()
#         nparams = len(params_upper_bnd)
        
#         # Initialize global parameters
#         params_ranges_half = (np.array(params_upper_bnd) - np.array(params_lower_bnd)) / 2
        
#         global_stds = np.random.uniform(low = 0.001,
#                                         high = params_ranges_half / 10,
#                                         size = (self.config['nparamsets'], nparams))
#         global_means = np.random.uniform(low = np.array(params_lower_bnd) + (params_ranges_half / 5),
#                                          high = np.array(params_upper_bnd) - (params_ranges_half / 5),
#                                          size = (self.config['nparamsets'], nparams))

#         # Initialize local parameters (by condition)
#         subject_param_grid = np.zeros((self.config['nparamsets'], self.config['nsubjects'], nparams))
        
#         for n in range(self.config['nparamsets']):
#             for i in range(self.config['nsubjects']):
#                 a, b = (np.array(params_lower_bnd) - global_means[n]) / global_stds[n], (np.array(params_upper_bnd) - global_means[n]) / global_stds[n]
#                 subject_param_grid[n, i, :] = np.float32(global_means[n] + truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[n])
                
# #                 Print statements to test if sampling from truncated distribution works properly
# #                 print('random variates')
# #                 print(truncnorm.rvs(a, b, size = global_stds.shape[1]))
# #                 print('samples')
# #                 print(subject_param_grid[n, i, :])

#         return subject_param_grid, global_stds, global_means


   
#     def generate_data_grid_hierarchical_parallel(self, 
#                                                  param_grid = []):
        
#         nparams = param_grid.shape[2]
#         args_list = self.make_args_starmap_ready(param_grid = np.reshape(param_grid, (-1, nparams)))

#         if self.config['n_cpus'] == 'all':
#             n_cpus = psutil.cpu_count(logical = False)
#         else:
#             n_cpus = self.config['n_cpus']

#         # Run Data generation
#         with Pool(processes = n_cpus) as pool:
#             data_grid = np.array(pool.starmap(self.data_generator, args_list))

#         data_grid = np.reshape(data_grid, (self.config['nparamsets'], self.config['nsubjects'], self.config['nsamples'], self.config['nchoices']))
    
#         return data_grid

#     def make_dataset_parameter_recovery_hierarchical(self,
#                                                      save = True):

#         param_grid, global_stds, global_means = self.make_param_grid_hierarchical()
        
#         self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'] * self.config['nsubjects'])]
        
#         if self.config['binned']:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nsubjects'], # add to config
#                                   self.config['nbins'],
#                                   self.config['nchoices']))
#         else:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nsubjects'],
#                                   self.config['nsamples'],
#                                   2))
        
#         for rep in range(self.config['nreps']):
#             # TD: ADD generate_data_grid_parallel_multisubject
#             data_grid[rep] = np.array(self.generate_data_grid_hierarchical_parallel(param_grid = param_grid))
#             print(rep, ' repetition finished') 
        
#         if save:
#             training_data_folder = self.method_folder + 'parameter_recovery_hierarchical_data_binned_' + str(int(self.config['binned'])) + \
#                                    '_nbins_' + str(self.config['nbins']) + \
#                                    '_n_' + str(self.config['nsamples'])
#             if not os.path.exists(training_data_folder):
#                 os.makedirs(training_data_folder)
            
#             print('saving dataset as ', training_data_folder + '/' + \
#                                         self.method + \
#                                         '_nchoices_' + str(self.config['nchoices']) + \
#                                         '_parameter_recovery_hierarchical_' + \
#                                         'binned_' + str(int(self.config['binned'])) + \
#                                         '_nbins_' + str(self.config['nbins']) + \
#                                         '_nreps_' + str(self.config['nreps']) + \
#                                         '_n_' + str(self.config['nsamples']) + \
#                                         '_nsubj_' + str(self.config['nsubjects']) + \
#                                         '.pickle')
            
#             meta = self.dgp_hyperparameters.copy()
#             if 'boundary' in meta.keys():
#                 del meta['boundary']

#             pickle.dump(([param_grid, global_stds, global_means], data_grid, meta), 
#                         open(training_data_folder + '/' + \
#                             self.method + \
#                             '_nchoices_' + str(self.config['nchoices']) + \
#                             '_parameter_recovery_hierarchical_' + \
#                             'binned_' + str(int(self.config['binned'])) + \
#                             '_nbins_' + str(self.config['nbins']) + \
#                             '_nreps_' + str(self.config['nreps']) + \
#                             '_n_' + str(self.config['nsamples']) + \
#                             '_nsubj_' + str(self.config['nsubjects']) + \
#                             '.pickle', 'wb'), 
#                         protocol = self.config['pickleprotocol'])
            
#             return 'Dataset completed'
#         else:
#             return ([param_grid, global_stds, global_means], data_grid, meta)

#     def make_dataset_r_sim(self,
#                            n_sim_bnds = [10000, 100000],
#                            save = True):
        
#         param_grid = self.make_param_grid_uniform()
#         self.nsamples = np.random.uniform(size = self.config['nparamsets'], 
#                                           high = n_sim_bnds[1],
#                                           low = n_sim_bnds[0])
        
#         if self.config['binned']:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nbins'],
#                                   self.config['nchoices']))
#         else:
#             return 'random number of simulations is supported under BINNED DATA only for now'
        
#         for rep in range(self.config['nreps']):
#             data_grid[rep] = np.array(self.generate_data_grid_parallel(param_grid = param_grid))
#             print(rep, ' repetition finished')
            
            
#         if save == True:
#             print('saving dataset')
#             pickle.dump((param_grid, data_grid), open(self.method_comparison_folder + \
#                                                       'base_data_uniform_r_sim_' + \
#                                                       str(n_sim_bnds[0]) + '_' + str(n_sim_bnds[1]) + \
#                                                       '_nreps_' + str(self.config['nreps']) + \
#                                                       '_' + self.file_id + '.pickle', 'wb'))
#             return 'Dataset completed'
#         else:
#             return param_grid, data_grid, self.nsamples
# ---------------------------------------------------------------------------------------

# # Functions outside the data generator class that use it --------------------------------
# def make_dataset_r_dgp(dgp_list = ['ddm', 'ornstein', 'angle', 'weibull', 'full_ddm'],
#                        machine = 'x7',
#                        r_nsamples = True,
#                        n_sim_bnds = [100, 100000],
#                        file_id = 'TEST',
#                        max_t = 10.0,
#                        delta_t = 0.001,
#                        config = None,
#                        save = False):
#     """Generates datasets across kinds of simulators
    
#     Parameter 
#     ---------
#     dgp_list : list
#         List of simulators that you would like to include (names match the simulators stored in kde_info.py)
#     machine : str
#         The machine the code is run on (basically changes folder directory for imports, meant to be temporary)
#     file_id : str
#         Attach an identifier to file name that is stored if 'save' is True (some file name formatting is already applied)
#     save : bool
#         If true saves outputs to pickle file
    
#     Returns
#     -------
#     list 
#         [0] list of arrays of parameters by dgp (n_dgp, nparamsets, n_parameters_given_dgp)
#         [1] list of arrays storing sampler outputs as histograms (n_repitions, n_parameters_sets, nbins, nchoices)
#         [2] array of model ids (dgp ids)
#         [3] dgp_list
        
#     """
  
#     if config['binned']:
#         model_ids = np.random.choice(len(dgp_list),
#                                      size = (config['nparamsets'], 1))
#         model_ids.sort(axis = 0)
#         data_grid_out = []
#         param_grid_out = []
#         nsamples_out = []
#     else:
#         return 'Config should specify binned = True for simulations to start'

#     for i in range(len(dgp_list)):
#         nparamsets = np.sum(model_ids == i)
        
#         # Change config to update the sampler
#         config['method'] = dgp_list[i]
#         # Change config to update the number of parameter sets we want to run for the current sampler
#         config['nparamsets'] = nparamsets
        
#         # Initialize data_generator class with new properties
#         dg_tmp = data_generator(machine = machine,
#                                 config = config,
#                                 max_t = max_t,
#                                 delta_t = delta_t)
        
#         # Run the simulator
#         if r_nsamples:
#             param_grid, data_grid, nsamples = dg_tmp.make_dataset_r_sim(n_sim_bnds = n_sim_bnds,
#                                                                         save = False)
#         else:
#             param_grid, data_grid = dg_tmp.make_dataset_train_network_unif(save = False)
            
#         print(data_grid.shape)
#         print(param_grid.shape)
        
#         # Append results
#         data_grid_out.append(data_grid)
#         param_grid_out.append(param_grid)
#         if r_nsamples:
#             nsamples_out.append(nsamples_out)

#     if save:
#         print('saving dataset')
#         if machine == 'x7':
#             out_folder = '/media/data_cifs/afengler/data/kde/rdgp/'
#         if machine == 'ccv':
#             out_folder = '/users/afengler/data/kde/rdgp/'
#         if machine == 'home':
#             out_folder = '/Users/afengler/OneDrive/project_nn_likelihoods/data/kde/rdgp/'
#         if machine == 'other':
#             if not os.path.exists('data_storage'):
#                 os.makedirs('data_storage')
#                 os.makedirs('data_storage/rdgp')
            
#             out_folder = 'data_storage/rdgp/'
            
#         out_folder = out_folder + 'training_data_' + str(int(config['binned'])) + \
#                      '_nbins_' + str(config['nbins']) + \
#                      '_nsimbnds_' + str(config['nsimbnds'][0]) + '_' + str(config['nsimbnds'][1]) +'/'
            
#         if not os.path.exists(out_folder):
#                 os.makedirs(out_folder)
                
#         pickle.dump((np.concatenate(data_grid_out, axis = 1), model_ids, param_grid_out, dgp_list),
#                                                    open(out_folder + \
#                                                    'rdgp_nchoices_' + str(config['nchoices']) + \
#                                                    '_nreps_' + str(config['nreps']) + \
#                                                    '_nsimbnds_' + str(config['nsimbnds'][0]) + '_' + \
#                                                    str(config['nsimbnds'][1]) + \
#                                                    '_' + str(config['file_id']) + '.pickle', 'wb'))
#         return 'Dataset completed'
#     else:
#         return (np.concatenate(data_grid_out, axis = 1), model_ids, param_grid_out, dgp_list)  

#      if args.datatype == 'cnn_train':
#         simulator = bs.simulator()
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)
#         out = dg.make_dataset_train_network_unif(save = args.save)
        
#     if args.datatype == 'parameter_recovery':
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)
#         out = dg.make_dataset_parameter_recovery(save = args.save)
    
#     if args.datatype == 'perturbation_experiment':      
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)
#         out = dg.make_dataset_perturbation_experiment(save = args.save)

#     if args.datatype == 'r_sim':
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)
#         out = dg.make_dataset_r_sim(n_sim_bnds = [100, 200000],
#                                         save = args.save)
        
#     if args.datatype == 'r_dgp':
#         out = make_dataset_r_dgp(dgp_list = args.dgplist,
#                                  machine = args.machine,
#                                  file_id = args.fileid,
#                                  r_nsamples = True,
#                                  n_sim_bnds = args.nsimbnds,
#                                  max_t = args.maxt,
#                                  delta_t = args.deltat,
#                                  config = config,
#                                  save = args.save)

#     if args.datatype == 'parameter_recovery_hierarchical':
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)

#         out = dg.make_dataset_parameter_recovery_hierarchical()


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
import sys
import argparse

# My own code
import kde_class as kde
#import cddm_data_simulation as ddm_simulator 
import boundary_functions as bf
import kde_training_utilities as kde_utils

if __name__ == "__main__":
    # Interface ------
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'ccv')
    CLI.add_argument("--method",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--simfolder",
                     type = str,
                     default = 'base_simulations')
    CLI.add_argument("--fileprefix",
                     type = str,
                     default = 'ddm_base_simulations')
    CLI.add_argument("--fileid",
                     type = str,
                     default = 'TEST')
    args = CLI.parse_args()
    print(args)
    
    # Specify base simulation folder ------
    if args.machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle",
                                         "rb"))[args.method]
        base_simulation_folder = method_params['method_folder_x7'] + args.simfolder +'/'
        
    if args.machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", 
                                         "rb"))[args.method]
        base_simulation_folder = method_params['method_folder'] + args.simfolder + '/'
        
    # FILTERS: GENERAL
    filters = {'mode': 20, # != 
               'choice_cnt': 10, # > 
               'mean_rt': 18, # < 
               'std': 0, # > 
               'mode_cnt_rel': 0.5  # < 
              }
    
    # Run filter new
    start_time = time.time()
    kde_utils.filter_simulations_fast(base_simulation_folder = base_simulation_folder,
                                      file_name_prefix = args.fileprefix,
                                      file_id = args.fileid,
                                      method_params = method_params,
                                      param_ranges = 'none',
                                      filters = filters)
    
    end_time = time.time()
    exec_time = end_time - start_time
    print('Time elapsed: ', exec_time)

    
    
    #import ddm_data_simulation as ddm_sim
import scipy as scp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import multiprocessing as mp
from  multiprocessing import Process
from  multiprocessing import Pool
import psutil
import pickle
import os
import time
import sys
import argparse

import kde_training_utilities as kde_util
import kde_class as kde

if __name__ == "__main__":
    # Interfact ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'ccv')
    CLI.add_argument('--method',
                     type = str,
                     default = 'ddm')
    CLI.add_argument('--simfolder',
                     type = str,
                     default = 'base_simulations')
    CLI.add_argument('--fileprefix',
                     type = str,
                     default = 'ddm_base_simulations')
    CLI.add_argument('--fileid',
                     type = str,
                     default = 'TEST')
    CLI.add_argument('--outfolder',
                     type = str,
                     default = 'train_test')
    CLI.add_argument('--nbyparam',
                     type = int,
                     default = 1000)
    CLI.add_argument('--mixture',
                     nargs = '*',
                     type = float,
                     default = [0.8, 0.1, 0.1])
    CLI.add_argument('--nproc',
                    type = int,
                    default = 8)
    CLI.add_argument('--analytic',
                     type = int,
                     default = 0)
    
    args = CLI.parse_args()
    print(args)
    
    # Specify base simulation folder ------
    if args.machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle",
                                         "rb"))[args.method]
        method_folder = method_params['method_folder_x7']

    if args.machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", 
                                         "rb"))[args.method]
        method_folder = method_params['method_folder']
    
    # Speficy names of process parameters
    process_params = method_params['param_names'] + method_params['boundary_param_names']
    
    # Make output folder if it doesn't exist
    if not os.path.isdir(method_folder + args.outfolder + '/'):
        os.mkdir(method_folder + args.outfolder + '/')
    
# STANDARD VERSION ----------------------------------------------------------------------------------------
    
    # Main function 
    start_time = time.time()
    kde_util.kde_from_simulations_fast_parallel(base_simulation_folder = method_folder + args.simfolder,
                                                file_name_prefix = args.fileprefix,
                                                file_id = args.fileid,
                                                target_folder = method_folder + args.outfolder,
                                                n_by_param = args.nbyparam,
                                                mixture_p = args.mixture,
                                                process_params = process_params,
                                                print_info = False,
                                                n_processes= args.nproc,
                                                analytic = args.analytic)
    
    end_time = time.time()
    exec_time = end_time - start_time
    print('Time elapsed: ', exec_time)

#-----------------------------------------------------------------------------------------------------------

# UNUSED --------------------------
    # LBA
#     process_params = ['v_0', 'v_1', 'A', 'b', 's', 'ndt']
#     files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

#     # DDM NDT
#     process_params = ['v', 'a', 'w', 'ndt']
#     files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

    # DDM ANGLE NDT
    #process_params = ['v', 'a', 'w', 'ndt', 'theta']
    #files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

   # print(mp.get_all_start_methods())
    
    
# # ALTERNATIVE VERSION

# # We should be able to parallelize this !
    
#     # Parallel
#     if args.nproc == 'all':
#         n_cpus = psutil.cpu_count(logical = False)
#     else:
#         n_cpus = args.nproc

#     print('Number of cpus: ')
#     print(n_cpus)
    
#     file_ = pickle.load(open( method_folder + args.simfolder + '/' + args.fileprefix + '_' + str(args.fileid) + '.pickle', 'rb' ) )
    
#     stat_ = pickle.load(open( method_folder + args.simfolder + '/simulator_statistics' + '_' + str(args.fileid) + '.pickle', 'rb' ) )
   
#     # Initializations
#     n_kde = int(args.nbyparam * args.mixture[0])
#     n_unif_down = int(args.nbyparam * args.mixture[1])
#     n_unif_up = int(args.nbyparam * args.mixture[2])
#     n_kde = n_kde + (args.nbyparam - n_kde - n_unif_up - n_unif_down) # correct n_kde if sum != args.nbyparam
    
#     # Add possible choices to file_[2] which is the meta data for the simulator (expected when loaded the kde class)
    
#     # TODO: THIS INFORMATION SHOULD BE INCLUDED AS META-DATA INTO THE BASE SIMULATOIN FILES
#     file_[2]['possible_choices'] = np.unique([-1, 1])
#     #file_[2]['possible_choices'] = np.unique(file_[1][0, :, 1])
#     file_[2]['possible_choices'].sort()

#     # CONTINUE HERE   
#     # Preparation loop --------------------------------------------------------------------
#     #s_id_kde = np.sum(stat_['keep_file']) * (n_unif_down + n_unif_up)
#     cnt = 0
#     starmap_iterator = ()
#     tmp_sim_data_ok = 0
#     results = []
#     for i in range(file_[1].shape[0]):
#         if stat_['keep_file'][i]:
            
#             # Don't remember what this part is doing....
#             if tmp_sim_data_ok:
#                 pass
#             else:
#                 tmp_sim_data = file_[1][i]
#                 tmp_sim_data_ok = 1
                
#             lb = cnt * (n_unif_down + n_unif_up + n_kde)

#             # Allocate to starmap tuple for mixture component 3
#             if args.analytic:
#                 starmap_iterator += ((file_[1][i, :, :].copy(), file_[0][i, :].copy(), file_[2].copy(), n_kde, n_unif_up, n_unif_down, cnt), )
#             else:
#                 starmap_iterator += ((file_[1][i, :, :], file_[2], n_kde, n_unif_up, n_unif_down, cnt), ) 
#                 #starmap_iterator += ((n_kde, n_unif_up, n_unif_down, cnt), )
#             # alternative
#             # tmp = i
#             # starmap_iterator += ((tmp), )
            
#             cnt += 1
#             if (cnt % 100 == 0) or (i == file_[1].shape[0] - 1):
#                 with Pool(processes = n_cpus, maxtasksperchild = 200) as pool:
#                     results.append(np.array(pool.starmap(kde_util.make_kde_data, starmap_iterator)).reshape((-1, 3)))   #.reshape((-1, 3))
#                     #result = pool.starmap(make_kde_data, starmap_iterator)
#                 starmap_iterator = ()
#                 print(i, 'arguments generated')
     
#     # Make dataframe to save
#     # Initialize dataframe
    
#     my_columns = process_params + ['rt', 'choice', 'log_l']
#     data = pd.DataFrame(np.zeros((np.sum(stat_['keep_file']) * args.nbyparam, len(my_columns))),
#                         columns = my_columns)    
    
#     #data.values[: , -3:] = result.reshape((-1, 3))
    
#     data.values[:, -3:] = np.concatenate(results)
#     # Filling in training data frame ---------------------------------------------------
#     cnt = 0
#     tmp_sim_data_ok = 0
#     for i in range(file_[1].shape[0]):
#         if stat_['keep_file'][i]:
            
#             # Don't remember what this part is doing....
#             if tmp_sim_data_ok:
#                 pass
#             else:
#                 tmp_sim_data = file_[1][i]
#                 tmp_sim_data_ok = 1
                
#             lb = cnt * (n_unif_down + n_unif_up + n_kde)

#             # Make empty dataframe of appropriate size
#             p_cnt = 0
            
#             for param in process_params:
#                 data.iloc[(lb):(lb + n_unif_down + n_unif_up + n_kde), my_columns.index(param)] = file_[0][i, p_cnt]
#                 p_cnt += 1
                
#             cnt += 1
#     # ----------------------------------------------------------------------------------

#     # Store data
#     print('writing data to file: ', method_folder + args.outfolder + '/data_' + str(args.fileid) + '.pickle')
#     pickle.dump(data.values, open(method_folder + args.outfolder + '/data_' + str(args.fileid) + '.pickle', 'wb'), protocol = 4)
    
#     # Write metafile if it doesn't exist already
#     # Hack for now: Just copy one of the base simulations files over
    
#     if os.path.isfile(method_folder + args.outfolder + '/meta_data.pickle'):
#         pass
#     else:
#         pickle.dump(tmp_sim_data, open(method_folder + args.outfolder + '/meta_data.pickle', 'wb') )

#     #return 0 #data
                                       