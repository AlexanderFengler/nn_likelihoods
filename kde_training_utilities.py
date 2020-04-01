# Basic python
import numpy as np
import scipy as scp
from scipy.stats import gamma
from scipy.stats import mode
from scipy.stats import itemfreq
from scipy.stats import mode
import pandas as pd
import random

# Parallelization
import multiprocessing as mp
from  multiprocessing import Process
from  multiprocessing import Pool
import psutil
import argparse

# System utilities
from datetime import datetime
import time
import os
import pickle
import uuid
import glob

# My own code
import kde_class
#import ddm_data_simulation as ddm_simulator
import boundary_functions as bf

# Plotting
import matplotlib.pyplot as plt

# /users/afengler/data/kde/full_ddm/training_data_binned_0_nbins_0_n_20000/full_ddm_nchoices_2_train_data_binned_0_nbins_0_n_20000_213.pickle
# /users/afengler/data/kde/full_ddm/training_data_binned_0_nbins_0_n_20000/simulator_statistics_213.pickle

def filter_simulations_fast(base_simulation_folder = '',
                            file_name_prefix = '',
                            file_id = 0,
                            method_params = [],
                            param_ranges = 'none', # either 'none' or dict that specifies allowed ranges for parameters
                            
                            filters = {'mode': 20, # != (checking if mode is max_rt)
                                       'choice_cnt': 10, # > (checking that each choice receive at least 10 samples in simulator)
                                       'mean_rt': 15, # < (checking that mean_rt is smaller than specified value
                                       'std': 0, # > (checking that std is positive for each choice)
                                       'mode_cnt_rel': 0.5  # < (checking that mode does not receive more than a proportion of samples for each choice)
                                 }
                      ):

    file_ = pickle.load(open( base_simulation_folder + file_name_prefix + '_' + str(file_id) + '.pickle', 'rb' ))
    init_cols = method_params['param_names'] + method_params['boundary_param_names']
    n_datasets = file_[1].shape[0]
    
    # Initialize data frame
    sim_stat_data = pd.DataFrame(file_[0], 
                                 columns = init_cols)
    
    # MAX RT BY SIMULATION: TEST SHOULD BE CONSISTENT
    n_simulations = file_[1].shape[1] #['n_samples']
    # TODO: BASE SIMULATIONS FILES NEED TO HOLD THE N-CHOICES PROPERTY DIRECTLY
    n_choices = 2
    #n_choices = len(np.unique(file_[1][0, :, 1])) # ['n_choices']
    
    # TODO: BASE SIMULATIONS NEED TO HOLD THE UNIQUE CHOICES PROPERTY DIRECTLY
    # RIGHT NOW THIS CODE USES THE DATA ITSELF TO RECOVER THE POSSIBLE CHOICES BUT THIS ALLOWS FOR READING IN N CHOICES < REAL N CHOICES
    choices = np.unique([-1, 1])
    #choices = np.unique(file_[1][0, :, 1])
    
    #n_choices = len(file_[0][2]['possible_choices'])
    #choices = file_[0][2]['possible_choices']
    
    max_rts = np.zeros((n_datasets, 1))
    
    max_t = file_[2]['max_t']
    sim_stat_data['max_t'] = max_t
    
    
    #max_ts[:] = max_t 
    max_ts = np.zeros((n_datasets, 1))
    stds = np.zeros((n_datasets, n_choices))
    mean_rts = np.zeros((n_datasets, n_choices))
    choice_cnts = np.zeros((n_datasets, n_choices))
    modes = np.zeros((n_datasets, n_choices))
    mode_cnts = np.zeros((n_datasets, n_choices))
    
    #sim_stat_data = [None] * n_datasets

    cnt = 0     
    for i in range(n_datasets):
        max_rts[i] = (file_[1][i, :, 0].max().round(2))
        max_ts[i] = max_t
        #max_ts[i] = (file_[1][i][2]['max_t'])
        # Standard deviation of reaction times
        choice_cnt = 0
        for choice_tmp in choices:

            tmp_rts = file_[1][i, :, 0][file_[1][i, :, 1] == choice_tmp]
            n_c = len(tmp_rts)
            choice_cnts[cnt, choice_cnt] = n_c
            mode_tmp = mode(tmp_rts)

            if n_c > 0:
                mean_rts[cnt, choice_cnt] = np.mean(tmp_rts)
                stds[cnt, choice_cnt] = np.std(tmp_rts)
                modes[cnt, choice_cnt] = float(mode_tmp[0])
                mode_cnts[cnt, choice_cnt] = int(mode_tmp[1])
            else:
                mean_rts[cnt, choice_cnt] = - 1
                stds[cnt, choice_cnt] = - 1
                modes[cnt, choice_cnt] = - 1
                mode_cnts[cnt, choice_cnt] = 0

            choice_cnt += 1

        # Basic data column 
        # TODO: Put this back in respecting new input format
        #sim_stat_data[cnt] = [file_[i][2][key] for key in list(file_[i][2].keys())]

        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)

    #sim_stat_data = pd.DataFrame(sim_stat_data, columns = file_[0][2].keys())
    # Compute some more columns
    for i in range(0, n_choices, 1):
        sim_stat_data['mean_rt_' + str(i)] = mean_rts[:, i]
        sim_stat_data['std_' + str(i)] = stds[:, i]
        sim_stat_data['choice_cnt_' + str(i)] = choice_cnts[:,i]
        sim_stat_data['mode_' + str(i)] = modes[:, i]
        sim_stat_data['mode_cnt_' + str(i)] = mode_cnts[:, i]

        # Derived Columns
        sim_stat_data['choice_prop_' + str(i)] = sim_stat_data['choice_cnt_' + str(i)] / n_simulations
        sim_stat_data['mode_cnt_rel_' + str(i)] = sim_stat_data['mode_cnt_' + str(i)] / sim_stat_data['choice_cnt_' + str(i)]

    # Clean-up
    sim_stat_data = sim_stat_data.round(decimals = 2)
    sim_stat_data = sim_stat_data.fillna(value = 0)

    # check that max_t is consistently the same value across simulations
    #assert len(np.unique(max_ts)) == 1

    # Now filtering

    # FILTER 1: PARAMETER RANGES
    if param_ranges == 'none':
            keep = sim_stat_data['max_t'] >= 0 # should return a vector of all true's
    else:
        cnt = 0
        for param in param_ranges.keys():
            if cnt == 0:
                keep = (sim_stat_data[param] >= param_ranges[param][0]) & (sim_stat_data[param] <= param_ranges[param][1])
            else:
                keep = (keep) & \
                       (sim_stat_data[param] >= param_ranges[param][0]) & (sim_stat_data[param] <= param_ranges[param][1])
            cnt += 1

    # FILTER 2: SANITY CHECKS (Filter-bank)
    for i in range(0, n_choices, 1):
        keep = (keep) & \
               (sim_stat_data['mode_' + str(i)] != filters['mode']) & \
               (sim_stat_data['choice_cnt_' + str(i)] > filters['choice_cnt']) & \
               (sim_stat_data['mean_rt_' + str(i)] < filters['mean_rt']) & \
               (sim_stat_data['std_' + str(i)] > filters['std']) & \
               (sim_stat_data['mode_cnt_rel_' + str(i)] < filters['mode_cnt_rel'])

    # Add keep_file column to
    sim_stat_data['keep_file'] = keep

    # Write files:
    #pickle.dump(list(sim_stat_data.loc[keep, 'file']), open(base_simulation_folder + '/keep_files.pickle', 'wb'))
    pickle.dump(sim_stat_data, 
                open(base_simulation_folder + '/simulator_statistics' + '_' + str(file_id) + '.pickle', 'wb'))
                     
    return sim_stat_data                     

def make_kde_data(data = [], metadata  = [], keep_file = 0, n_kde = 100, n_unif_up = 100, n_unif_down = 100, idx = 0):
    out = np.zeros((n_kde + n_unif_up + n_unif_down, 3))
    tmp_kde = kde_class.logkde((data[:, 0], data[:, 1], metadata))
    
    # Get kde part
    samples_kde = tmp_kde.kde_sample(n_samples = n_kde)
    likelihoods_kde = tmp_kde.kde_eval(data = samples_kde).ravel()
    
    out[:n_kde, 0] = samples_kde[0].ravel()
    out[:n_kde, 1] = samples_kde[1].ravel()
    out[:n_kde, 2] = likelihoods_kde

    # Get positive uniform part:
    choice_tmp = np.random.choice(metadata['possible_choices'], size = n_unif_up)

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
    choice_tmp = np.random.choice(metadata['possible_choices'], #['possible_choices'],
                                    size = n_unif_down)
    
    rt_tmp = np.random.uniform(low = - 1,
                                high = 0.0001,
                                size = n_unif_down)

    out[(n_kde + n_unif_up):, 0] = rt_tmp
    out[(n_kde + n_unif_up):, 1] = choice_tmp
    out[(n_kde + n_unif_up):, 2] = -66.77497
    if idx % 10 == 0:
        print(idx)
    return out

# We should be able to parallelize this !
def kde_from_simulations_fast_parallel(base_simulation_folder = '',
                                       file_name_prefix = '',
                                       file_id = 1,
                                       target_folder = '', 
                                       n_by_param = 3000,
                                       mixture_p = [0.8, 0.1, 0.1],
                                       process_params = ['v', 'a', 'w', 'c1', 'c2'],
                                       print_info = False
                                       ):

    file_ = pickle.load(open( base_simulation_folder + '/' + file_name_prefix + '_' + str(file_id) + '.pickle', 'rb' ) )
    stat_ = pickle.load(open( base_simulation_folder + '/simulator_statistics' + '_' + str(file_id) + '.pickle', 'rb' ) )

    # Initialize dataframe
    my_columns = process_params + ['rt', 'choice', 'log_l']
    data = pd.DataFrame(np.zeros((np.sum(stat_['keep_file']) * n_by_param, len(my_columns))),
                        columns = my_columns)             
     
    n_kde = int(n_by_param * mixture_p[0])
    n_unif_down = int(n_by_param * mixture_p[1])
    n_unif_up = int(n_by_param * mixture_p[2])
    n_kde = n_kde + (n_by_param - n_kde - n_unif_up - n_unif_down) # correct n_kde if sum != n_by_param
    
    # Add possible choices to file_[2] which is the meta data for the simulator (expected when loaded the kde class)
    
    # TODO: THIS INFORMATION SHOULD BE INCLUDED AS META-DATA INTO THE BASE SIMULATOIN FILES
    file_[2]['possible_choices'] = np.unique([-1,1])
    #file_[2]['possible_choices'] = np.unique(file_[1][0, :, 1])
    file_[2]['possible_choices'].sort()

    # CONTINUE HERE   
    # Main while loop --------------------------------------------------------------------
    #row_cnt = 0
    s_id_kde = np.sum(stat_['keep_file']) * (n_unif_down + n_unif_up)
    cnt = 0
    starmap_iterator = ()
    for i in range(file_[1].shape[0]):
        if stat_['keep_file'][i]:
            tmp_sim_data = file_[1][i]
            lb = cnt * (n_unif_down + n_unif_up + n_kde)
            lb_kde = s_id_kde + (cnt * (n_kde))
            
            # Make empty dataframe of appropriate size
            p_cnt = 0
            
            for param in process_params:
                data.iloc[(lb):(lb + n_unif_down + n_unif_up + n_kde), my_columns.index(param)] = file_[0][i, p_cnt] #tmp_sim_data[2][param]
                #data.iloc[(lb_kde):(lb_kde + n_kde), my_columns.index(param)] = file_[0][i, p_cnt]
                p_cnt += 1
            
            # Allocate to starmap tuple for mixture component 3
            starmap_iterator += ((file_[1][i, :, :], file_[2], stat_['keep_file'][i], n_kde, n_unif_up, n_unif_down, cnt), )
            
            cnt += 1
            if i % 100 == 0:
                print(i, 'unif part generated')
    
    # MIXTURE COMPONENT 3: KDE DATA ----------------------------------------------------
    # Parallel
    n_cpus = psutil.cpu_count(logical = False)
    with Pool(processes = n_cpus, maxtasksperchild=100) as pool:
        # data.iloc[s_id_kde: , ['rt', 'choice', 'log_l']]
        data.iloc[: , -3:] = np.array(pool.starmap(make_kde_data, starmap_iterator)).reshape((-1, 3))
        print(data)
        #print(type(out_))
        #print(out_)
        #print(out_.shape)
    # ----------------------------------------------------------------------------------

    # Store data
    
    print('writing data to file: ', target_folder + '/data_' + str(file_id) + '.pickle')
    pickle.dump(data.values, open(target_folder + '/data_' + str(file_id) + '.pickle', 'wb'), protocol = 4)
    
    #data.to_pickle(target_folder + '/data_' + str(file_id) + '.pickle' , protocol = 4)

    # Write metafile if it doesn't exist already
    # Hack for now: Just copy one of the base simulations files over
    if os.path.isfile(target_folder + '/meta_data.pickle'):
        pass
    else:
        pickle.dump(tmp_sim_data, open(target_folder + '/meta_data.pickle', 'wb') )

    return data

def kde_from_simulations_fast(base_simulation_folder = '',
                              file_name_prefix = '',
                              file_id = 1,
                              target_folder = '',
                              n_by_param = 3000,
                              mixture_p = [0.8, 0.1, 0.1],
                              process_params = ['v', 'a', 'w', 'c1', 'c2'],
                              print_info = False
                             ):
    
    file_ = pickle.load(open( base_simulation_folder + '/' + file_name_prefix + '_' + str(file_id) + '.pickle', 'rb' ) )
    stat_ = pickle.load(open( base_simulation_folder + '/simulator_statistics' + '_' + str(file_id) + '.pickle', 'rb' ) )

    # Initialize dataframe
    my_columns = process_params + ['rt', 'choice', 'log_l']
    data = pd.DataFrame(np.zeros((np.sum(stat_['keep_file']) * n_by_param, len(my_columns))),
                        columns = my_columns)             
     
    n_kde = int(n_by_param * mixture_p[0])
    n_unif_down = int(n_by_param * mixture_p[1])
    n_unif_up = int(n_by_param * mixture_p[2])
    n_kde = n_kde + (n_by_param - n_kde - n_unif_up - n_unif_down) # correct n_kde if sum != n_by_param
    
    # Add possible choices to file_[2] which is the meta data for the simulator (expected when loaded the kde class)
    
    # TODO: THIS INFORMATION SHOULD BE INCLUDED AS META-DATA INTO THE BASE SIMULATOIN FILES
    file_[2]['possible_choices'] = np.unique([-1,1])
    #file_[2]['possible_choices'] = np.unique(file_[1][0, :, 1])
    file_[2]['possible_choices'].sort()
    # CONTINUE HERE   
    # Main while loop --------------------------------------------------------------------
    #row_cnt = 0
    cnt = 0
    for i in range(file_[1].shape[0]):
        if stat_['keep_file'][i]:
            # Read in simulator file
            tmp_sim_data = file_[1][i]
            lb = cnt * n_by_param
            
            # Make empty dataframe of appropriate size
            p_cnt = 0
            for param in process_params:
                data.iloc[(lb):(lb + n_by_param), my_columns.index(param)] = file_[0][i, p_cnt] #tmp_sim_data[2][param]
                p_cnt += 1
            
            # MIXTURE COMPONENT 1: Get simulated data from kde -------------------------------
            tmp_kde = kde_class.logkde((file_[1][i, :, 0], file_[1][i, :, 1], file_[2])) #[tmp_sim_data)
            tmp_kde_samples = tmp_kde.kde_sample(n_samples = n_kde)

            data.iloc[lb:(lb + n_kde), my_columns.index('rt')] = tmp_kde_samples[0].ravel()
            data.iloc[lb:(lb + n_kde), my_columns.index('choice')] = tmp_kde_samples[1].ravel()
            data.iloc[lb:(lb + n_kde), my_columns.index('log_l')] = tmp_kde.kde_eval(data = tmp_kde_samples).ravel()
            # --------------------------------------------------------------------------------

            # MIXTURE COMPONENT 2: Negative uniform part -------------------------------------
            choice_tmp = np.random.choice(file_[2]['possible_choices'], #['possible_choices'],
                                          size = n_unif_down)
            
            rt_tmp = np.random.uniform(low = - 1,
                                       high = 0.0001,
                                       size = n_unif_down)

            data.iloc[(lb + n_kde):(lb + n_kde + n_unif_down), my_columns.index('rt')] = rt_tmp
            data.iloc[(lb + n_kde):(lb + n_kde + n_unif_down), my_columns.index('choice')] = choice_tmp
            data.iloc[(lb + n_kde):(lb + n_kde + n_unif_down), my_columns.index('log_l')] = -66.77497 # the number corresponds to log(1e-29)
            # ---------------------------------------------------------------------------------


            # MIXTURE COMPONENT 3: Positive uniform part --------------------------------------
            choice_tmp = np.random.choice(file_[2]['possible_choices'],
                                          size = n_unif_up)

            if file_[2]['max_t'] < 100:
                rt_tmp = np.random.uniform(low = 0.0001,
                                           high = file_[2]['max_t'],
                                           size = n_unif_up)
            else:
                rt_tmp = np.random.uniform(low = 0.0001, 
                                           high = 100,
                                           size = n_unif_up)

            data.iloc[(lb + n_kde + n_unif_down):(lb + n_by_param), my_columns.index('rt')] = rt_tmp
            data.iloc[(lb + n_kde + n_unif_down):(lb + n_by_param), my_columns.index('choice')] = choice_tmp
            data.iloc[(lb + n_kde + n_unif_down):(lb + n_by_param), my_columns.index('log_l')] = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp))
            # ----------------------------------------------------------------------------------
            cnt += 1
            if i % 10 == 0:
                print(i, 'kdes generated')
    # -----------------------------------------------------------------------------------

    # Store data
    print('writing data to file: ', target_folder + '/data_' + str(file_id) + '.pickle')
    pickle.dump(data.values, open(target_folder + '/data_' + str(file_id) + '.pickle', 'wb'), protocol = 4)
    #data.to_pickle(target_folder + '/data_' + str(file_id) + '.pickle' , protocol = 4)

    # Write metafile if it doesn't exist already
    # Hack for now: Just copy one of the base simulations files over
    if os.path.isfile(target_folder + '/meta_data.pickle'):
        pass
    else:
        pickle.dump(tmp_sim_data, open(target_folder + '/meta_data.pickle', 'wb') )

    return data

def kde_load_data_new(path = '',
                      file_id_list = '',
                      prelog_cutoff_low = 1e-29,
                      prelog_cutoff_high = 100,
                      n_samples_by_dataset = 10000000,
                      return_log = True):
    
    # Read in two datasets to get meta data for the subsequent
    print('Reading in initial dataset')
    tmp_data = np.load(path + 'data_' + str(file_id_list[0]) + '.pickle', allow_pickle = True)
    
#     features_init = np.load(path + 'feed_features_' + str(file_id_list[0]) + '.npy')
#     labels_init = np.load(path + 'feed_labels_' + str(file_id_list[0]) + '.npy')
    
#     features_init = tmp_data[]
#     labels_init = np.load(path + 'feed_labels_' + str(file_id_list[0]) + '.npy')
    
    # Collect some meta data 
    n_files = len(file_id_list)
    print('n_files: ', n_files)
    print('n_samples_by_dataset: ', n_samples_by_dataset)
    
    # Allocate memory for data  
    print('Allocating data arrays')
    features = np.zeros((n_files * n_samples_by_dataset, tmp_data.shape[1] - 1))
    labels = np.zeros((n_files * n_samples_by_dataset, 1))
    
    # Read in data of initialization files
    cnt_samples = tmp_data.shape[0]
    features[:cnt_samples, :] = tmp_data[:, :-1]
    labels[:cnt_samples, 0] = tmp_data[:, -1]
    
    # Read in remaining files into preallocated np.array
    for i in range(1, n_files, 1):
        tmp_data = np.load(path + 'data_' + str(file_id_list[i]) + '.pickle', allow_pickle = True)
        n_rows_tmp = tmp_data.shape[0]
        features[(cnt_samples): (cnt_samples + n_rows_tmp), :] = tmp_data[:, :-1]
        labels[(cnt_samples): (cnt_samples + n_rows_tmp), 0] = tmp_data[:, -1]
        cnt_samples += n_rows_tmp
        
        #features[(i * n_samples_by_dataset): ((i + 1) * (n_samples_by_dataset)), :] = np.load(path + 'feed_features_' + \
          #                                                                                    str(file_id_list[i]) + '.npy', allow_pickle = True)
        #labels[(i * n_samples_by_dataset): ((i + 1) * (n_samples_by_dataset)), :] = np.load(path + 'feed_labels_' + \
         #                                                                                  str(file_id_list[i]) + '.npy', allow_pickle = True)
        print(i, ' files processed')
        
    features.resize((cnt_samples, features.shape[1]), refcheck = False)
    labels.resize((cnt_samples, labels.shape[1]), refcheck = False)
    
    print('new n rows features: ', features.shape[0])
    print('new n rows labels: ', labels.shape[0])
    
    if prelog_cutoff_low != 'none':
        labels[labels < np.log(prelog_cutoff_low)] = np.log(prelog_cutoff_low)
    if prelog_cutoff_high != 'none':    
        labels[labels > np.log(prelog_cutoff_high)] = np.log(prelog_cutoff_high)
        
    if return_log == False:
        labels = np.exp(labels)
    
    return features, labels

# NO LONGER USED -------
# def filter_simulations(base_simulation_folder = '',
#                        param_ranges = 'none', # either 'none' or dict that specifies allowed ranges for parameters
#                        filters = {'mode': 20, # != (checking if mode is max_rt)
#                                   'choice_cnt': 10, # > (checking that each choice receive at least 10 samples in simulator)
#                                   'mean_rt': 15, # < (checking that mean_rt is smaller than specified value
#                                   'std': 0, # > (checking that std is positive for each choice)
#                                   'mode_cnt_rel': 0.5  # < (checking that mode does not receive more than a proportion of samples for each choice)
#                                  }
#                       ):

#     # Initialization
#     files_ = os.listdir(base_simulation_folder)

#     # Drop some files that are not simulations:
#     new_files_ = []

#     for file_ in files_:
#         if (file_ != 'keep_files.pickle') & (file_ != 'simulator_statistics.pickle'):
#             new_files_.append(file_)

#     files_ = new_files_

#     n_files = len(files_)
#     init_file = pickle.load(open( base_simulation_folder + files_[0], 'rb' ))
#     init_cols = list(init_file[2].keys())

#     # Initialize data frame
#     sim_stat_data = pd.DataFrame(np.zeros((n_files, len(init_cols))), columns = init_cols)

#     # MAX RT BY SIMULATION: TEST SHOULD BE CONSISTENT
#     n_simulations = init_file[2]['n_samples']
#     n_choices = len(init_file[2]['possible_choices'])

#     max_rts = []
#     max_ts = []

#     stds = np.zeros((n_files, n_choices))
#     mean_rts = np.zeros((n_files, n_choices))
#     choice_cnts = np.zeros((n_files, n_choices))
#     modes = np.zeros((n_files, n_choices))
#     mode_cnts = np.zeros((n_files, n_choices))

#     sim_stat_data = [None] * n_files

#     cnt = 0
#     for file_ in files_:
#         tmp = pickle.load(open( base_simulation_folder + file_ , 'rb'))
#         max_rts.append(tmp[0].max().round(2))
#         max_ts.append(tmp[2]['max_t'])

#         # Standard deviation of reaction times
#         choice_cnt = 0
#         for choice_tmp in tmp[2]['possible_choices']:

#             tmp_rts = tmp[0][tmp[1] == choice_tmp]
#             n_c = len(tmp_rts)
#             choice_cnts[cnt, choice_cnt] = n_c

#             mode_tmp = mode(tmp_rts)

#             if n_c > 0:
#                 mean_rts[cnt, choice_cnt] = np.mean(tmp_rts)
#                 stds[cnt, choice_cnt] = np.std(tmp_rts)
#                 modes[cnt, choice_cnt] = float(mode_tmp[0])
#                 mode_cnts[cnt, choice_cnt] = int(mode_tmp[1])
#             else:
#                 mean_rts[cnt, choice_cnt] = -1
#                 stds[cnt, choice_cnt] = -1
#                 modes[cnt, choice_cnt] = -1
#                 mode_cnts[cnt, choice_cnt] = 0

#             choice_cnt += 1

#         # Basic data column
#         sim_stat_data[cnt] = [tmp[2][key] for key in list(tmp[2].keys())]

#         cnt += 1
#         if cnt % 1000 == 0:
#             print(cnt)

#     sim_stat_data = pd.DataFrame(sim_stat_data, columns = init_file[2].keys())

#     # Compute some more columns
#     for i in range(0, n_choices, 1):
#         sim_stat_data['mean_rt_' + str(i)] = mean_rts[:, i]
#         sim_stat_data['std_' + str(i)] = stds[:, i]
#         sim_stat_data['choice_cnt_' + str(i)] = choice_cnts[:,i]
#         sim_stat_data['mode_' + str(i)] = modes[:, i]
#         sim_stat_data['mode_cnt_' + str(i)] = mode_cnts[:, i]

#         # Derived Columns
#         sim_stat_data['choice_prop_' + str(i)] = sim_stat_data['choice_cnt_' + str(i)] / n_simulations
#         sim_stat_data['mode_cnt_rel_' + str(i)] = sim_stat_data['mode_cnt_' + str(i)] / sim_stat_data['choice_cnt_' + str(i)]

#     # Add a columns that stores files
#     sim_stat_data['file'] = files_

#     # Clean-up
#     sim_stat_data = sim_stat_data.round(decimals = 2)
#     sim_stat_data = sim_stat_data.fillna(value = 0)

#     # check that max_t is consistently the same value across simulations
#     assert len(np.unique(max_ts)) == 1

#     # check that max_rt is <= max_t + 0.00001 (adding for rounding)
#     # SKIPPING BECAUSE OF INCONSISTENCY PROBLEMS WITH NDT --
#     #assert max(max_rts) <= np.unique(max_ts)[0] + 0.0001

#     # Now filtering

#     # FILTER 1: PARAMETER RANGES
#     if param_ranges == 'none':
#             keep = sim_stat_data['a'] >= 0 # should return a vector of all true's
#     else:
#         cnt = 0
#         for param in param_ranges.keys():
#             if cnt == 0:
#                 keep = (sim_stat_data[param] >= param_ranges[param][0]) & (sim_stat_data[param] <= param_ranges[param][1])
#             else:
#                 keep = (keep) & \
#                        (sim_stat_data[param] >= param_ranges[param][0]) & (sim_stat_data[param] <= param_ranges[param][1])
#             cnt += 1

#     # FILTER 2: SANITY CHECKS (Filter-bank)
#     for i in range(0, n_choices, 1):
#         keep = (keep) & \
#                (sim_stat_data['mode_' + str(i)] != filters['mode']) & \
#                (sim_stat_data['choice_cnt_' + str(i)] > filters['choice_cnt']) & \
#                (sim_stat_data['mean_rt_' + str(i)] < filters['mean_rt']) & \
#                (sim_stat_data['std_' + str(i)] > filters['std']) & \
#                (sim_stat_data['mode_cnt_rel_' + str(i)] < filters['mode_cnt_rel'])

#     # Add keep_file column to
#     sim_stat_data['keep_file'] = keep

#     # Write files:
#     pickle.dump(list(sim_stat_data.loc[keep, 'file']), open(base_simulation_folder + '/keep_files.pickle', 'wb'))
#     pickle.dump(sim_stat_data, open(base_simulation_folder + '/simulator_statistics.pickle', 'wb'))

#     return sim_stat_data


# def kde_from_simulations(base_simulation_folder = '',
#                          target_folder = '',
#                          n_total = 10000,
#                          mixture_p = [0.8, 0.1, 0.1], # maybe here I can instead pass a function that provides a sampler
#                          process_params = ['v', 'a', 'w', 'c1', 'c2'],
#                          print_info = False,
#                          files_ = 'all', # either 'all' or list of files
#                          p_files = 0.2):

#     # Get files if files_ is not specified as a list
#     if files_ == 'all':
#         files_ = os.listdir(base_simulation_folder)

#     n_files = int(len(files_) * p_files)

#     # Get sample of files
#     files_ = random.sample(files_, n_files)

#     # Make every string full directory
#     files_ = [base_simulation_folder + '/' + file_ for file_ in files_]

#     # Compute some derived quantities
#     n_samples_by_kde_tmp = int(n_total / n_files)

#     n_kde = int(n_samples_by_kde_tmp * mixture_p[0])
#     n_unif_up = int(n_samples_by_kde_tmp * mixture_p[1])
#     n_unif_down = int(n_samples_by_kde_tmp * mixture_p[2])

#     n_samples_by_kde = n_kde + n_unif_up + n_unif_down
#     n_samples_total = n_samples_by_kde * n_files

#     # Generate basic empty dataframe (do more clever)
#     my_columns = process_params + ['rt', 'choice', 'log_l']

#     # Initialize dataframe
#     data = pd.DataFrame(np.zeros((n_samples_total, len(my_columns))),
#                         columns = my_columns)

#     # Main while loop --------------------------------------------------------------------
#     row_cnt = 0
#     cnt = 0
#     for file_ in files_:
#         # Read in simulator file
#         tmp_sim_data = pickle.load( open( file_ , "rb" ) )

#         # Make empty dataframe of appropriate size
#         for param in process_params:
#             data.iloc[row_cnt:(row_cnt + n_samples_by_kde), my_columns.index(param)] = tmp_sim_data[2][param]

#         # MIXTURE COMPONENT 1: Get simulated data from kde -------------------------------
#         tmp_kde = kde_class.logkde(tmp_sim_data)
#         tmp_kde_samples = tmp_kde.kde_sample(n_samples = n_kde)

#         data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('rt')] = tmp_kde_samples[0].ravel()
#         data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('choice')] = tmp_kde_samples[1].ravel()
#         data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('log_l')] = tmp_kde.kde_eval(data = tmp_kde_samples).ravel()
    
#     #--------------------------------------------------------------------------------

#         # MIXTURE COMPONENT 2: Negative uniform part -------------------------------------
#         choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'],
#                                       size = n_unif_down)

#         rt_tmp = np.random.uniform(low = - 1,
#                                    high = 0.0001,
#                                    size = n_unif_down)

#         data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('rt')] = rt_tmp
#         data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('choice')] = choice_tmp
#         data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('log_l')] = -66.77497 # the number corresponds to log(1e-29)
#         # ---------------------------------------------------------------------------------


#         # MIXTURE COMPONENT 3: Positive uniform part --------------------------------------
#         choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'],
#                                       size = n_unif_up)
        
#         if tmp_sim_data[2]['max_t'] < 100:
#             rt_tmp = np.random.uniform(low = 0.0001,
#                                        high = tmp_sim_data[2]['max_t'],
#                                        size = n_unif_up)
#         else:
#             rt_tmp = np.random.uniform(low = 0.0001, 
#                                        high = 100,
#                                        size = n_unif_up)

#         data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('rt')] = rt_tmp
#         data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('choice')] = choice_tmp
#         data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('log_l')] = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp))
#         # ----------------------------------------------------------------------------------

#         row_cnt += n_samples_by_kde
#         cnt += 1

#         if cnt % 1000 == 0:
#             print(cnt, 'kdes generated')
#     # -----------------------------------------------------------------------------------

#     # Store data
#     print('writing data to file')
#     data.to_pickle(target_folder + '/data_' + uuid.uuid1().hex + '.pickle' , protocol = 4)

#     # Write metafile if it doesn't exist already
#     # Hack for now: Just copy one of the base simulations files over
#     if os.path.isfile(target_folder + '/meta_data.pickle'):
#         pass
#     else:
#         pickle.dump(tmp_sim_data,  open(target_folder + '/meta_data.pickle', 'wb') )

#     return data


# def kde_load_data(folder = '',
#                   return_log = False, # function expects log_likelihood data so if log = false we take exponent
#                   prelog_cutoff = 1e-29 # either 'none' or number (like 1e-29)
#                   ):
    
#     # Load training data from file
#     train_features = pd.read_pickle(folder + '/train_features.pickle')
#     train_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/train_labels.pickle')))

#     # Load test data from file
#     test_features = pd.read_pickle(folder + '/test_features.pickle')
#     test_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/test_labels.pickle')))

#     # Preprocess labels
#     # 1. get rid of numbers smaller than log(1e-29)
#     if prelog_cutoff != 'none':
#         train_labels[train_labels < np.log(prelog_cutoff)] = np.log(prelog_cutoff)
#         test_labels[test_labels < np.log(prelog_cutoff)] = np.log(prelog_cutoff)

#     # 2. Take exp
#     if return_log == False:
#         train_labels = np.exp(train_labels)
#         test_labels = np.exp(test_labels)
        
#     return train_features, train_labels, test_features, test_labels   


# # TO DO: Specify a 'data_*' file-selection procedure
# def kde_make_train_data(path = '',
#                         n_files_out = 10,
#                         file_in_list = 'all' # provide as full directory or 'all'
#                        ):

#     # Get files in specified folder

#     # Check if folder currently contains a train-test split
#     #print('check if we have a train and test sets already')

#     # Read in and concatenate files
#     print('read, concatenate data...')
#     if file_in_list == 'all':
#         files_ = glob.glob(path + 'data_*')
#     else:
#         files_ = [file_ for file_ in file_in_list]
#         data = pd.concat([pd.read_pickle(file_) for file_ in files_])
                         
#     # Shuffle data
#     print('shuffle...')
#     np.random.shuffle(data.values)
#     data.reset_index(drop = True, 
#                      inplace = True)

#     # Get meta-data from dataframe
#     n_cols = len(list(data.keys()))

#     # Write to file
#     print('writing to file...')
    
#     n_samples_by_dataset = int(np.floor(data.shape[0] / n_files_out))
    
#     for i in range(n_files_out):
#         np.save(path + 'feed_features_' + str(i) + '.npy',
#                 data.iloc[(i * n_samples_by_dataset) : ((i + 1) * n_samples_by_dataset), :(n_cols - 1)].values)
        
#         np.save(path + 'feed_labels_' + str(i) + '.npy',
#                 np.expand_dims(data.iloc[(i * n_samples_by_dataset) : ((i + 1) * n_samples_by_dataset), (n_cols - 1)].values, axis = 1))
        
#         print(i, ' files written')
        
#     return 'success'
