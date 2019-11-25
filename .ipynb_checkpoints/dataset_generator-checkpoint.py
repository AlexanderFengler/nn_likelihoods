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
#from tqdm import tqdm

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from cdwiener import batch_fptd
import clba

# Parallelization
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Pool
import psutil
import argparse

# --------------------------------------------------------------------------------

# Data generator class that generates datasets for us ----------------------------
class data_generator():
    def __init__(self,
                 machine = 'x7',
                 file_id = 'id',
                 max_t = 20.0,
                 config = None):
    # INIT -----------------------------------------
        self.machine = machine
        self.file_id = file_id
        self.config = config
        self.method = self.config['method']
        
        if self.machine == 'x7':  
            self.method_params = pickle.load(open("/media/data_cifs/afengler/" + \
                                                  "git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[self.method]
            self.method_comparison_folder = self.method_params['output_folder_x7']
            self.method_folder = self.method_params['method_folder_x7']

        if self.machine == 'ccv':
            self.method_params = pickle.load(open("/users/afengler/git_repos/" + \
                                                  "nn_likelihoods/kde_stats.pickle", "rb"))[self.method]
            self.method_comparison_folder = self.method_params['output_folder']
            self.method_folder = self.method_params['method_folder']

        self.dgp_hyperparameters = dict(self.method_params['dgp_hyperparameters'])
        self.dgp_hyperparameters['max_t'] = max_t
        self.dgp_hyperparameters['n_samples'] = self.config['n_samples']
   # ----------------------------------------------------
   
    def data_generator(self, *args):
        # Get simulations             
        simulator_output = self.method_params['dgp'](*args)
        
        # Bin simulations if so specified in config
        if self.config['binned']:
            labels = self.bin_simulator_output(out = simulator_output,
                                               bin_dt = self.config['bin_dt'],
                                               n_bins = self.config['n_bins'])
            return labels
        # Return simulator output as [rts, choices] instead if we don't specify binned
        else:
            return np.concatenate([simulator_output[0], simulator_output[1]], axis = 1)
  
    def bin_simulator_output(self, 
                             out = [0, 0],
                             bin_dt = 0.04,
                             n_bins = 0): # ['v', 'a', 'w', 'ndt', 'angle']

        # Generate bins
        if n_bins == 0:
            n_bins = int(out[2]['max_t'] / bin_dt)
            bins = np.linspace(0, out[2]['max_t'], n_bins + 1)
        else:    
            bins = np.linspace(0, out[2]['max_t'], n_bins + 1)

        cnt = 0
        counts = np.zeros( (n_bins, len(out[2]['possible_choices']) ) )

        for choice in out[2]['possible_choices']:
            counts[:, cnt] = np.histogram(out[0][out[1] == choice], bins = bins)[0] / out[2]['n_samples']
            cnt += 1
        return counts
    
    def zip_dict(self,
                 x = [], 
                 key_vec = ['a', 'b', 'c']):
        return dict(zip(key_vec, x))
    
 
    def make_args_starmap_ready(self,
                                param_grid = []):
            
        n_parameter_sets = param_grid.shape[0]
        n_params = param_grid.shape[1]
        n_boundary_params = len(self.method_params['boundary_param_names'])
        n_process_params = len(self.method_params["param_names"])
        
        # Boundary parameter touples
        if n_boundary_params > 0:
            boundary_param_tuples = np.apply_along_axis(self.zip_dict, 1,  
                                                        param_grid[:, n_process_params:], 
                                                        self.method_params['boundary_param_names'])
        
        # Process parameters to make suitable for parallel processing
        if self.config['n_choices'] <= 2 and self.method != 'lba':
            process_param_tuples = tuple(map(tuple, param_grid[:, :n_process_params]))
        
        elif self.config['n_choices'] > 2 or self.method == 'lba':
            process_param_tuples = tuple()
            for i in range(param_grid.shape[0]):
                tuple_tmp = tuple()
                cnt = 0
                
                for j in range(len(self.method_params['param_names'])):
                    if self.method_params['param_depends_on_n_choice'][j]:
                        tuple_tmp += (param_grid[i, cnt: (cnt + self.config['n_choices'])], )
                        cnt += self.config['n_choices']
                    else:
                        tuple_tmp += (param_grid[i, cnt], )
                        cnt += 1
                process_param_tuples += (tuple_tmp, )
                
        # If models are lca or race we want pass noise standarad deviation as an array instead of a single value
        if self.method == 'race_model':
            self.dgp_hyperparameters['s'] = np.float32(np.repeat(self.dgp_hyperparameters['s'], self.config['n_choices']))
        
        # Make final list of tuples of parameters
        args_list = []
        for i in range(n_parameter_sets):
            process_params = process_param_tuples[i]
            
            if n_boundary_params > 0:
                boundary_params = (boundary_param_tuples[i], )
            else:
                boundary_params = ({},)
            
            # N samples
            self.dgp_hyperparameters['n_samples'] = self.n_samples[i]
            
            sampler_hyperparameters = tuple(self.dgp_hyperparameters.values())
            if self.method == 'lba': # TODO change lba sampler to accept boundary params?
                args_list.append(process_params + sampler_hyperparameters)
            else:
                args_list.append(process_params + sampler_hyperparameters + boundary_params)
                #print(process_params + sampler_hyperparameters + boundary_params)
        return args_list
    
    def clean_up_parameters(self):
        
        if self.config['mode'] == 'test':
            param_bounds = self.method_params['param_bounds_sampler'] + self.method_params['boundary_param_bounds_sampler']
        if self.config['mode'] == 'train':
            param_bounds = self.method_params['param_bounds_network'] + self.method_params['boundary_param_bounds_network']
        if self.config['mode'] == 'cnn':
            param_bounds = self.method_params['param_bounds_cnn'] + self.method_params['boundary_param_bounds_cnn']

        # If model is lba, lca, race we need to expand parameter boundaries to account for
        # parameters that depend on the number of choices
        if self.method == 'lba' or self.method == 'lca' or self.method == 'race_model':
            param_depends_on_n = self.method_params['param_depends_on_n_choice']
            param_bounds_tmp = []
            
            n_process_params = len(self.method_params['param_names'])
            
            p_cnt = 0
            for i in range(n_process_params):
                if self.method_params['param_depends_on_n_choice'][i]:
                    for c in range(self.config['n_choices']):
                        param_bounds_tmp.append(param_bounds[i])
                        p_cnt += 1
                else:
                    param_bounds_tmp.append(param_bounds[i])
                    p_cnt += 1
            
            self.method_params['n_process_parameters'] = p_cnt
                    
            param_bounds_tmp += param_bounds[n_process_params:]
            params_upper_bnd = [bnd[0] for bnd in param_bounds_tmp]
            params_lower_bnd = [bnd[1] for bnd in param_bounds_tmp]
            #print(params_lower_bnd)
            
            
        # If our model is not lba, race, lca we use simple procedure 
        else:
            params_upper_bnd = [bnd[0] for bnd in param_bounds]
            params_lower_bnd = [bnd[1] for bnd in param_bounds]
            
        return params_upper_bnd, params_lower_bnd
    
                
    def make_param_grid_perturbation_experiment(self):

        n_perturbation_levels = len(self.config['perturbation_sizes'][0])
        n_params = len(self.method_params['param_names']) + len(self.method_params['boundary_param_names'])
        
        # Get parameter bounds
        params_upper_bnd, params_lower_bnd = self.clean_up_parameters()
        n_params = len(params_upper_bnd)
        experiment_row_cnt = (n_params * n_perturbation_levels) + 1
                       
        meta_dat = pd.DataFrame(np.zeros((experiment_row_cnt, 2)), 
                                columns = ['param', 'perturbation_level']) 
                       
        param_grid = np.zeros((self.config['n_experiments'], experiment_row_cnt, n_params))
        
        # Make the parameter grids
        for i in range(self.config['n_experiments']):
                       
            # Reinitialize row cnt 
            cnt = 0
            
            # Get base parameters for perturbation experiment i
            param_grid_tmp = np.float32(np.random.uniform(low = params_lower_bnd, 
                                                          high = params_upper_bnd), dtype = np.float32)
                       
            # Store as first row in experiment i
            param_grid[i, cnt, :] = param_grid_tmp

            # Store meta data for experiment (only need to do once --> i ==0)
            if i == 0:
                meta_dat.loc[cnt, :] = [-1, -1]

            cnt += 1
                       
            # Fill in perturbation experiment data i
            for p in range(n_params):
                for l in range(n_perturbation_levels):
                    param_grid_perturbed = param_grid_tmp.copy()
                    if param_grid_tmp[p] > ((params_upper_bnd[p] - params_lower_bnd[p]) / 2):
                        param_grid_perturbed[p] -= self.config['perturbation_sizes'][p][l]
                    else:
                        param_grid_perturbed[p] += self.config['perturbation_sizes'][p][l]

                    param_grid[i, cnt, :] = param_grid_perturbed

                    if i  == 0:
                        meta_dat.loc[cnt, :] = [int(p), int(l)]

                    cnt += 1 
                       
        return (param_grid, meta_dat)
                       
    def make_param_grid_uniform(self,
                                n_parameter_sets = None):
        
        if n_parameter_sets == None:
            n_parameter_sets = self.config['n_parameter_sets']
            
        # Initializations
        params_upper_bnd, params_lower_bnd = self.clean_up_parameters()
        n_params = len(params_upper_bnd)
        param_grid = np.zeros((n_parameter_sets, n_params), dtype = np.float32)
        
        # Generate parameters
        param_grid[:, :] = np.float32(np.random.uniform(low = params_lower_bnd,
                                                        high = params_upper_bnd,
                                                        size = (n_parameter_sets, n_params)))
        return param_grid 
    
    def generate_data_grid_parallel(self,
                                    param_grid = []):
        
        args_list = self.make_args_starmap_ready(param_grid = param_grid)
        
        if self.config['n_cpus'] == 'all':
            n_cpus = psutil.cpu_count(logical = False)
        else:
            n_cpus = self.config['n_cpus']

        # Run Data generation
        with Pool(processes = n_cpus) as pool:
            data_grid = np.array(pool.starmap(self.data_generator, args_list))

        return data_grid     
        
    def make_dataset_perturbation_experiment(self,
                                             save = True):
        
        param_grid, meta_dat = self.param_grid_perturbation_experiment()
             
        if self.config['binned']:           
            data_grid = np.zeros((self.config['n_reps'],
                                  self.config['n_experiments'],
                                  param_grid.shape[1], 
                                  self.config['n_bins'],
                                  self.config['n_choices']))         
        
        else:
            data_grid = np.zeros((self.config['n_reps'],
                                  self.config['n_experiments'],
                                  param_grid.shape[1], 
                                  self.config['n_samples'],
                                  2)) 

        for experiment in range(self.config['n_experiments']):
            for rep in range(self.config['n_reps']):
                data_grid[rep, experiment] = self.generate_data_grid_parallel(param_grid = param_grid[experiment])
                print(experiment, ' experiment data finished')
        
        if save == True:
            print('saving dataset')
            pickle.dump((param_grid, data_grid, meta_dat), open(self.method_comparison_folder + \
                                                                'base_data_perturbation_experiment_nexp_' + \
                                                                str(self.config['n_experiments']) + \
                                                                '_nreps_' + str(self.config['nreps']) + \
                                                                '_n_' + str(self.config['n_samples']) + \
                                                                '_' + self.file_id + '.pickle', 'wb'))
                        
            return 'Dataset completed'
        else:
            return param_grid, data_grid, meta_dat
    
    def make_dataset_uniform(self,
                             save = True):
        
        param_grid = self.make_param_grid_uniform()
        self.n_samples = [self.config['n_samples'] for i in range(self.config['n_parameter_sets'])]
        
        if self.config['binned']:
            data_grid = np.zeros((self.config['n_reps'],
                                  self.config['n_parameter_sets'],
                                  self.config['n_bins'],
                                  self.config['n_choices']))
        else:
            data_grid = np.zeros((self.config['n_reps'],
                                  self.config['n_parameter_sets'],
                                  self.config['n_samples'],
                                  2))
        
        for rep in range(self.config['n_reps']):
            data_grid[rep] = np.array(self.generate_data_grid_parallel(param_grid = param_grid))
            print(rep, ' repetition finished') 
        
        if save == True:
            print('saving dataset as ', self.method_comparison_folder + \
                                        'base_data_uniform_' + \
                                        '_npar_' + str(self.config['n_parameter_sets']) + \
                                        '_nreps_' + str(self.config['n_reps']) + \
                                        '_n_' + str(self.config['n_samples']) + \
                                        '_' + self.file_id + '.pickle')
            
            pickle.dump((param_grid, data_grid, self.dgp_hyperparameters), open(self.method_comparison_folder + \
                                                                              'base_data_uniform' + \
                                                                              '_npar_' + str(self.config['n_parameter_sets']) + \
                                                                              '_nreps_' + str(self.config['nreps']) + \
                                                                              '_n_' + str(self.config['n_samples']) + \
                                                                              '_' + self.file_id + '.pickle', 'wb'))
            return 'Dataset completed'
        else:
            return param_grid, data_grid
    
    def make_dataset_train_network_unif(self,
                                        save = True):

        param_grid = self.make_param_grid_uniform()
        self.n_samples = [self.config['n_samples'] for i in range(self.config['n_parameter_sets'])]

        if self.config['binned']:
            data_grid = np.zeros((self.config['n_parameter_sets'],
                                  self.config['n_bins'],
                                  self.config['n_choices']))
        else:
            data_grid = np.zeros((self.config['n_parameter_sets'],
                                  self.config['n_samples'],
                                  2))
        #print(param_grid)
        data_grid = np.array(self.generate_data_grid_parallel(param_grid = param_grid))

        if save:
            training_data_folder = self.method_folder + 'training_data_binned_' + str(int(self.config['binned'])) + \
                                   '_nbins_' + str(self.config['n_bins']) + \
                                   '_n_' + str(self.config['n_samples'])
            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)
            
            print('saving dataset as ', training_data_folder + '/' + \
                                        self.method + '_train_data_' + \
                                        'binned_' + str(int(self.config['binned'])) + \
                                        '_nbins_' + str(self.config['n_bins']) + \
                                        '_n_' + str(self.config['n_samples']) + \
                                        '_' + self.file_id + '.pickle')
            
            meta = self.dgp_hyperparameters.copy()
            if 'boundary' in meta.keys():
                meta.pop('boundary')

            pickle.dump((param_grid, data_grid, meta), open(training_data_folder + '/' + \
                                                            self.method + \
                                                            '_nchoices_' + str(self.config['n_choices']) + \
                                                            '_train_data_' + \
                                                            'binned_' + str(int(self.config['binned'])) + \
                                                            '_nbins_' + str(self.config['n_bins']) + \
                                                            '_n_' + str(self.config['n_samples']) + \
                                                            '_' + self.file_id + '.pickle', 'wb'), protocol = 2)
            return 'Dataset completed'
        else:
            return param_grid, data_grid
    
    def make_dataset_r_sim(self,
                           n_sim_bnds = [10000, 100000],
                           save = True):
        
        param_grid = self.make_param_grid_uniform()
        self.n_samples = np.random.uniform(size = self.config['n_parameter_sets'], 
                                               high = n_sim_bnds[1],
                                               low = n_sim_bnds[0])
        
        if self.config['binned']:
            data_grid = np.zeros((self.config['n_reps'],
                                  self.config['n_parameter_sets'],
                                  self.config['n_bins'],
                                  self.config['n_choices']))
        else:
            return 'random number of simulations is supported under BINNED DATA only for now'
        
        for rep in range(self.config['n_reps']):
            data_grid[rep] = np.array(self.generate_data_grid_parallel(param_grid = param_grid))
            print(rep, ' repetition finished')
            
            
        if save == True:
            print('saving dataset')
            pickle.dump((param_grid, data_grid), open(self.method_comparison_folder + \
                                                      'base_data_uniform_r_sim_' + \
                                                      str(n_sim_bnds[0]) + '_' + str(n_sim_bnds[1]) + \
                                                      '_n_reps_' + str(self.config['nreps']) + \
                                                      '_' + self.file_id + '.pickle', 'wb'))
            return 'Dataset completed'
        else:
            return param_grid, data_grid
# ---------------------------------------------------------------------------------------

# Functions outside the data generator class that use it --------------------------------
def make_dataset_r_dgp(dgp_list = ['ddm', 'ornstein', 'angle', 'weibull'],
                       machine = 'x7',
                       file_id = 'TEST',
                       max_rt = 20.0,
                       config = None,
                       save = False):
    """Generates datasets across kinds of simulators
    
    Parameter 
    ---------
    dgp_list : list
        List of simulators that you would like to include (names match the simulators stored in kde_info.py)
    machine : str
        The machine the code is run on (basically changes folder directory for imports, meant to be temporary)
    file_id : str
        Attach an identifier to file name that is stored if 'save' is True (some file name formatting is already applied)
    save : bool
        If true saves outputs to pickle file
    
    Returns
    -------
    list 
        [0] list of arrays of parameters by dgp (n_dgp, n_parameter_sets, n_parameters_given_dgp)
        [1] list of arrays storing sampler outputs as histograms (n_repitions, n_parameters_sets, n_bins, n_choices)
        [2] array of model ids (dgp ids)
        [3] dgp_list
        
    """
  
    if config['binned']:
        model_ids = np.random.choice(len(dgp_list), size = (config['n_parameter_sets'], 1))
        model_ids.sort()
        data_grid_out = []
        param_grid_out = []
    else:
        return 'Config should specify binned = True for simulations to start'

    for i in range(len(dgp_list)):
        n_parameter_sets = np.sum(model_ids == i)
        
        # Change config to update the sampler
        config['method'] = dgp_list[i]
        # Change config to update the number of parameter sets we want to run for the current sampler
        config['n_parameter_sets'] = n_parameter_sets
        
        # Initialize data_generator class with new properties
        dg_tmp = data_generator(machine = machine,
                                config = config,
                                max_rt = max_rt)
        # Run the simulator
        data_grid, param_grid = dg_tmp.make_dataset_uniform(save = False)
        
        # Append results
        data_grid_out.append(data_grid)
        param_grid_out.append(param_grid)

    if save:
        print('saving dataset')
        pickle.dump((np.concatenate(out), model_ids, dgp_list), open(method_comparison_folder + \
                                                                'base_data_uniform_r_dgp' + \
                                                                '_n_reps_' + str(self.config['nreps']) + \
                                                                '_n_' + str(self.config['n_samples']) + \
                                                                '_' + self.file_id + '.pickle', 'wb'))
        return 'Dataset completed'
    
    else:
        return (data_grid_out, param_grid_out, model_ids, dgp_list)  

def bin_arbitrary_fptd(out = [0, 0],
                       bin_dt = 0.04,
                       n_bins = 256,
                       n_choices = 2,
                       choice_codes = [-1.0, 1.0],
                       max_t = 20.0,
                       n_samples = 20000): # ['v', 'a', 'w', 'ndt', 'angle']

    # Generate bins
    if n_bins == 0:
        n_bins = int(max_t / bin_dt)
        bins = np.linspace(0, max_t, n_bins + 1)
    else:    
        bins = np.linspace(0, max_t, n_bins + 1)

    cnt = 0
    counts = np.zeros( (n_bins, n_choices) ) 

    for choice in choice_codes:
        counts[:, cnt] = np.histogram(out[:, 0][out[:, 1] == choice], bins = bins)[0] 
        cnt += 1
    return counts
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
                     default = ['ddm', 'ornstein', 'angle', 'weibull'])
    CLI.add_argument("--datatype",
                     type = str,
                     default = 'uniform') # 'uniform', 'perturbation_experiment', 'r_sim', 'r_dgp', 'train_network'
    CLI.add_argument("--binned",
                     type = bool,
                     default = 1)
    CLI.add_argument("--nbins",
                     type = int,
                     default = 256)
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 20000)
    CLI.add_argument("--nchoices",
                     type = int,
                     default = 2)
    CLI.add_argument("--mode",
                     type = str,
                     default = 'train') # train, test, cnn
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
    
    args = CLI.parse_args()
    print(args)
    
    machine = args.machine
    
    # Load basic config data
    if machine == 'x7':
        config = yaml.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/config_files/config_data_generator.yaml"))
        
    if machine == 'ccv':
        config = yaml.load(open("/users/afengler/git_repos/nn_likelihoods/config_files/config_data_generator.yaml"))
     
    # Update config with specifics of run
    config['method'] = args.dgplist[0]
    config['mode'] = args.mode
    config['file_id'] = args.fileid
    config['n_samples'] = args.nsamples
    config['binned'] = args.binned
    config['n_bins'] = args.nbins
    #config['type_of_experiment'] = args.datatype
    config['n_choices'] = args.nchoices
    print(config['n_choices'])
    config['n_parameter_sets'] = args.nparamsets
    
    # Get data for the type of dataset we want
    if args.datatype == 'cnn_train':
        dg = data_generator(machine = args.machine,
                            file_id = args.fileid,
                            max_t = args.maxt,
                            config = config)
        out = dg.make_dataset_train_network_unif(save = args.save)
    if args.datatype == 'uniform':
        dg = data_generator(machine = args.machine,
                             file_id = args.fileid,
                             max_t = args.maxt,
                             config = config)
        out = dg.make_dataset_uniform(save = args.save)
    
    if args.datatype == 'perturbation_experiment':      
        dg = data_generator(machine = args.machine,
                            file_id = args.fileid,
                            max_t = args.maxt,
                            config = config)
        out = dg_tmp.make_dataset_perturbation_experiment(save = args.save)

    if args.datatype == 'r_sim':
        dg = data_generator(machine = args.machine,
                            file_id = args.fileid,
                            max_t = args.maxt,
                            config = config)
        out = dg_tmp.make_dataset_r_sim(n_sim_bnds = [100, 200000],
                                  save = args.save)
        
    if args.datatype == 'r_dgp':
        out = make_dataset_r_dgp(dgp_list = args.dgplist,
                                 machine = args.machine,
                                 file_id = args.fileid,
                                 max_t = args.maxt,
                                 config = config,
                                 save = args.save)
    print('Finished')