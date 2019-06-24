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

# System utilities
from datetime import datetime
import time
import os
import pickle
import uuid

# My own code
import kde_class
import ddm_data_simulation as ddm_simulator
import boundary_functions as bf

# Plotting
import matplotlib.pyplot as plt

def filter_simulations(base_simulation_folder = '',
                       param_ranges = 'none', # either 'none' or dict that specifies allowed ranges for parameters
                       filters = {'mode': 20, # != (checking if mode is max_rt)
                                  'choice_cnt': 10, # > (checking that each choice receive at least 10 samples in simulator)
                                  'mean_rt': 15, # < (checking that mean_rt is smaller than specified value
                                  'std': 0, # > (checking that std is positive for each choice)
                                  'mode_cnt_rel': 0.5  # < (checking that mode does not receive more than a proportion of samples for each choice)
                                 }
                      ):

    # Initialization
    files_ = os.listdir(base_simulation_folder)

    # Drop some files that are not simulations:
    new_files_ = []

    for file_ in files_:
        if (file_ != 'keep_files.pickle') & (file_ != 'simulator_statistics.pickle'):
            new_files_.append(file_)

    files_ = new_files_

    n_files = len(files_)
    init_file = pickle.load(open( base_simulation_folder + files_[0], 'rb' ))
    init_cols = list(init_file[2].keys())

    # Initialize data frame
    sim_stat_data = pd.DataFrame(np.zeros((n_files, len(init_cols))), columns = init_cols)

    # MAX RT BY SIMULATION: TEST SHOULD BE CONSISTENT
    n_simulations = init_file[2]['n_samples']
    n_choices = len(init_file[2]['possible_choices'])

    max_rts = []
    max_ts = []

    stds = np.zeros((n_files, n_choices))
    mean_rts = np.zeros((n_files, n_choices))
    choice_cnts = np.zeros((n_files, n_choices))
    modes = np.zeros((n_files, n_choices))
    mode_cnts = np.zeros((n_files, n_choices))

    sim_stat_data = [None] * n_files

    cnt = 0
    for file_ in files_:
        tmp = pickle.load(open( base_simulation_folder + file_ , 'rb'))
        max_rts.append(tmp[0].max().round(2))
        max_ts.append(tmp[2]['max_t'])

        # Standard deviation of reaction times
        choice_cnt = 0
        for choice_tmp in tmp[2]['possible_choices']:

            tmp_rts = tmp[0][tmp[1] == choice_tmp]
            n_c = len(tmp_rts)
            choice_cnts[cnt, choice_cnt] = n_c

            mode_tmp = mode(tmp_rts)

            if n_c > 0:
                mean_rts[cnt, choice_cnt] = np.mean(tmp_rts)
                stds[cnt, choice_cnt] = np.std(tmp_rts)
                modes[cnt, choice_cnt] = float(mode_tmp[0])
                mode_cnts[cnt, choice_cnt] = int(mode_tmp[1])
            else:
                mean_rts[cnt, choice_cnt] = -1
                stds[cnt, choice_cnt] = -1
                modes[cnt, choice_cnt] = -1
                mode_cnts[cnt, choice_cnt] = 0

            choice_cnt += 1

        # Basic data column
        sim_stat_data[cnt] = [tmp[2][key] for key in list(tmp[2].keys())]

        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)

    sim_stat_data = pd.DataFrame(sim_stat_data, columns = init_file[2].keys())

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

    # Add a columns that stores files
    sim_stat_data['file'] = files_

    # Clean-up
    sim_stat_data = sim_stat_data.round(decimals = 2)
    sim_stat_data = sim_stat_data.fillna(value = 0)

    # check that max_t is consistently the same value across simulations
    assert len(np.unique(max_ts)) == 1

    # check that max_rt is <= max_t + 0.00001 (adding for rounding)
    assert max(max_rts) <= np.unique(max_ts)[0] + 0.0001

    # Now filtering

    # FILTER 1: PARAMETER RANGES
    if param_ranges == 'none':
            keep = sim_stat_data['a'] >= 0 # should return a vector of all true's
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
    pickle.dump(list(sim_stat_data.loc[keep, 'file']), open(base_simulation_folder + '/keep_files.pickle', 'wb'))
    pickle.dump(sim_stat_data, open(base_simulation_folder + '/simulator_statistics.pickle', 'wb'))

    return sim_stat_data


def kde_from_simulations(base_simulation_folder = '',
                         target_folder = '',
                         n_total = 10000,
                         mixture_p = [0.8, 0.1, 0.1], # maybe here I can instead pass a function that provides a sampler
                         process_params = ['v', 'a', 'w', 'c1', 'c2'],
                         print_info = False,
                         files_ = 'all', # either 'all' or list of files
                         p_files = 0.2):

    # Get files if files_ is not specified as a list
    if files_ == 'all':
        files_ = os.listdir(base_simulation_folder)

    n_files = int(len(files_) * p_files)

    # Get sample of files
    files_ = random.sample(files_, n_files)

    # Make every string full directory
    files_ = [base_simulation_folder + '/' + file_ for file_ in files_]


    # Compute some derived quantities
    n_samples_by_kde_tmp = int(n_total / n_files)

    n_kde = int(n_samples_by_kde_tmp * mixture_p[0])
    n_unif_up = int(n_samples_by_kde_tmp * mixture_p[1])
    n_unif_down = int(n_samples_by_kde_tmp * mixture_p[2])

    n_samples_by_kde = n_kde + n_unif_up + n_unif_down
    n_samples_total = n_samples_by_kde * n_files

    # Generate basic empty dataframe (do more clever)
    my_columns = process_params + ['rt', 'choice', 'log_l']

    # Initialize dataframe
    data = pd.DataFrame(np.zeros((n_samples_total, len(my_columns))),
                        columns = my_columns)

    # Main while loop --------------------------------------------------------------------
    row_cnt = 0
    cnt = 0
    for file_ in files_:
        # Read in simulator file
        tmp_sim_data = pickle.load( open( file_ , "rb" ) )

        # Make empty dataframe of appropriate size
        for param in process_params:
            data.iloc[row_cnt:(row_cnt + n_samples_by_kde), my_columns.index(param)] = tmp_sim_data[2][param]

        # MIXTURE COMPONENT 1: Get simulated data from kde -------------------------------
        tmp_kde = kde_class.logkde(tmp_sim_data)
        tmp_kde_samples = tmp_kde.kde_sample(n_samples = n_kde)

        data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('rt')] = tmp_kde_samples[0].ravel()
        data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('choice')] = tmp_kde_samples[1].ravel()
        data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('log_l')] = tmp_kde.kde_eval(data = tmp_kde_samples).ravel()
        # --------------------------------------------------------------------------------

        # MIXTURE COMPONENT 2: Negative uniform part -------------------------------------
        choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'],
                                      size = n_unif_down)

        rt_tmp = np.random.uniform(low = - 1,
                                   high = 0.0001,
                                   size = n_unif_down)

        data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('rt')] = rt_tmp
        data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('choice')] = choice_tmp
        data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('log_l')] = -66.77497 # the number corresponds to log(1e-29)
        # ---------------------------------------------------------------------------------


        # MIXTURE COMPONENT 3: Positive uniform part --------------------------------------
        choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'],
                                      size = n_unif_up)

        rt_tmp = np.random.uniform(low = 0.0001,
                                   high = tmp_sim_data[2]['max_t'],
                                   size = n_unif_up)

        data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('rt')] = rt_tmp
        data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('choice')] = choice_tmp
        data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('log_l')] = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp))
        # ----------------------------------------------------------------------------------

        row_cnt += n_samples_by_kde
        cnt += 1

        if cnt % 1000 == 0:
            print(cnt, 'kdes generated')
    # -----------------------------------------------------------------------------------

    # Store data
    print('writing data to file')
    data.to_pickle(target_folder + '/data_' + uuid.uuid1().hex + '.pickle' , protocol = 4)

    # Write metafile if it doesn't exist already
    # Hack for now: Just copy one of the base simulations files over
    if os.path.isfile(target_folder + '/meta_data.pickle'):
        pass
    else:
        pickle.dump(tmp_sim_data,  open(target_folder + '/meta_data.pickle', 'wb') )

    return data

# TO DO: Specify a 'data_*' file-selection procedure
def kde_make_train_test_split(folder = '',
                              p_train = 0.8):

    # Get files in specified folder
    print('get files in folder')
    files_ = os.listdir(folder)

    # Check if folder currently contains a train-test split
    print('check if we have a train and test sets already')
    for file_ in files_:
        if file_[:7] == 'train_f':
            return 'looks like a train test split exists in folder: Please remove before running this function'

    # If no train-test split in folder: collect 'data_*' files
    print('folder clean so proceeding...')
    data_files = []
    for file_ in files_:
        if file_[:5] == 'data_':
            data_files.append(file_)

    # Read in and concatenate files
    print('read, concatenate and shuffle data')
    data = pd.concat([pd.read_pickle(folder + file_) for file_ in data_files])

    # Shuffle data
    np.random.shuffle(data.values)
    data.reset_index(drop = True, inplace = True)

    # Get meta-data from dataframe
    n_cols = len(list(data.keys()))

    # Get train and test ids
    print('get training and test indices')
    train_id = np.random.choice(a = [True, False],
                                size = data.shape[0],
                                replace = True,
                                p = [p_train, 1 - p_train])

    test_id = np.invert(train_id)

    # Write to file
    print('writing to file...')
    data.iloc[train_id, :(n_cols - 1)].to_pickle(folder + 'train_features.pickle',
                                                          protocol = 4)

    data.iloc[test_id, :(n_cols - 1)].to_pickle(folder + 'test_features.pickle',
                                                         protocol = 4)

    data.iloc[train_id, (n_cols - 1)].to_pickle(folder + 'train_labels.pickle',
                                                         protocol = 4)

    data.iloc[test_id, (n_cols - 1)].to_pickle(folder + 'test_labels.pickle',
                                                        protocol = 4)

    return 'success'

def kde_load_data(folder = '',
                  log = False,
                  prelog_cutoff = 1e-29 # either 'none' or number (like 1e-29)
                  ):

    # Load training data from file
    train_features = pd.read_pickle(folder + '/train_features.pickle')
    train_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/train_labels.pickle')))

    # Load test data from file
    test_features = pd.read_pickle(folder + '/test_features.pickle')
    test_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/test_labels.pickle')))

    # Preprocess labels
    # 1. get rid of numbers smaller than log(1e-29)
    if prelog_cutoff != 'none':
        train_labels[train_labels < np.log(prelog_cutoff)] = np.log(prelog_cutoff)
        test_labels[test_labels < np.log(prelog_cutoff)] = np.log(prelog_cutoff)

    # 2. Take exp
    if log == False:
        train_labels = np.exp(train_labels)
        test_labels = np.exp(test_labels)

    return train_features, train_labels, test_features, test_labels
