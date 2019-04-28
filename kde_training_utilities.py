# Basic python 
import numpy as np
import scipy as scp
import pandas as pd

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

def generate_training_base_kde(file_name = '',
                               from_folder = '',
                               to_folder = '',
                               n_samples_by_kde = 10000,
                               mixture_p = [0.8, 0.1, 0.1], # [proportion from process, proportion low uniform, proportion high uniform]
                               model = 'ddm',
                               print_info = False,
                               target_file_format = 'pickle' # at this point can be 'pickle', 'csv'
                               ): 
    
    # currently just first 10 files for testing
    
    # REWRITE HERE: INSTEAD OF FOR LOOP DEFINE THIS FUNCTION SUCH THAT IT EXECUTES FOR A SINGLE FILE ONLY
    # 'fid',
    my_columns = ['v', 'a', 'w', 'rt', 'choice', 'log_l', 'kde_type', 'bandwidth_rule' , 'mixture', 'model']
    tmp_data = pd.DataFrame(np.zeros((n_samples_by_kde, len(my_columns))),
                            columns = my_columns)
    
    n_kde = int(n_samples_by_kde * mixture_p[0])
    n_unif_up = int(n_samples_by_kde * mixture_p[1])
    n_unif_down = int(n_samples_by_kde * mixture_p[2])
    tmp_sim_data = pickle.load( open( from_folder + '/' + file_name, "rb" ) )
    
    tmp_data.iloc[:, my_columns.index('v')] = tmp_sim_data[2]['v'] 
    tmp_data.iloc[:, my_columns.index('a')] = tmp_sim_data[2]['a']
    tmp_data.iloc[:, my_columns.index('w')] = tmp_sim_data[2]['w']
    tmp_data.iloc[:, my_columns.index('kde_type')] = 'gaussian'
    tmp_data.iloc[:, my_columns.index('bandwidth_rule')] = 'silverman'
    tmp_data.iloc[:, my_columns.index('mixture')] = str(mixture_p)
    tmp_data.iloc[:, my_columns.index('model')] = model
    #tmp_data.iloc[:, my_columns.index('fid')] = file_name
    
    # Get simulated data from kde
    tmp_kde = kde_class.logkde(tmp_sim_data)
    tmp_kde_samples = tmp_kde.kde_sample(n_samples = n_kde)

    tmp_data.iloc[:n_kde, my_columns.index('rt')] = tmp_kde_samples[0]
    tmp_data.iloc[:n_kde, my_columns.index('choice')] = tmp_kde_samples[1]
    tmp_data.iloc[:n_kde, my_columns.index('log_l')] = tmp_kde.kde_eval(data = tmp_kde_samples)

    # Generate rest of the data (to make sure we observe outliers that are negative and close to max_rt)
    # Negative uniform part
    choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'], 
                                  size = n_unif_down)
    rt_tmp = np.random.uniform(low = -1, 
                               high = 0.0001, 
                               size = n_unif_down)

    tmp_data.iloc[n_kde:(n_kde + n_unif_down), my_columns.index('rt')] = rt_tmp
    tmp_data.iloc[n_kde:(n_kde + n_unif_down), my_columns.index('choice')] = choice_tmp
    tmp_data.iloc[n_kde:(n_kde + n_unif_down), my_columns.index('log_l')] = -66.77497 # the number corresponds to log(1e-29)

    # Positive uniform part
    choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'], 
                                  size = n_unif_up)
    rt_tmp = np.random.uniform(low = 0.0001, 
                               high = tmp_sim_data[2]['max_t'], 
                               size = n_unif_up)

    tmp_data.iloc[(n_kde + n_unif_down): , my_columns.index('rt')] = rt_tmp
    tmp_data.iloc[(n_kde + n_unif_down): , my_columns.index('choice')] = choice_tmp
    tmp_data.iloc[(n_kde + n_unif_down):, my_columns.index('log_l')] = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp))
    
    # Some code to get the exact right files name
    # TO BE FIXED : (uuid needs to be separated from 'seconds' component of file name in the 'base_simulations' folder
    
    # Write to file
    if target_file_format == 'csv':
        tmp_data.to_csv(to_folder + '/kde_' + model + '_' + file_name[-39:-7] + '.csv', index = False)
    if target_file_format == 'pickle':
        tmp_data.to_pickle(to_folder + '/kde_' + model + '_' + file_name[-39:-7] + '.pickle')
    
    # Print statements
    print('Finished ' + file_name)
    print('Files generated at this point: ', len(os.listdir(to_folder)))

def generate_training_base_kde_flexbound(file_name = '',
                                           from_folder = '',
                                           to_folder = '',
                                           n_samples_by_kde = 10000,
                                           mixture_p = [0.8, 0.1, 0.1], # [proportion from process, proportion low uniform, proportion high uniform]
                                           model = 'ddm_flexbound',
                                           print_info = False,
                                           target_file_format = 'pickle' # at this point can be 'pickle', 'csv'
                                           ): 
    
    # currently just first 10 files for testing
    
    # REWRITE HERE: INSTEAD OF FOR LOOP DEFINE THIS FUNCTION SUCH THAT IT EXECUTES FOR A SINGLE FILE ONLY
    # 'fid',
    my_columns = ['v', 'a', 'w', 'c1', 'c2', 'rt', 'choice', 'log_l', 'kde_type', 'bandwidth_rule' , 'mixture', 'model']
    tmp_data = pd.DataFrame(np.zeros((n_samples_by_kde, len(my_columns))),
                            columns = my_columns)
    
    n_kde = int(n_samples_by_kde * mixture_p[0])
    n_unif_up = int(n_samples_by_kde * mixture_p[1])
    n_unif_down = int(n_samples_by_kde * mixture_p[2])
    tmp_sim_data = pickle.load( open( from_folder + '/' + file_name, "rb" ) )
    
    tmp_data.iloc[:, my_columns.index('v')] = tmp_sim_data[2]['v'] 
    tmp_data.iloc[:, my_columns.index('a')] = tmp_sim_data[2]['a']
    tmp_data.iloc[:, my_columns.index('w')] = tmp_sim_data[2]['w']
    tmp_data.iloc[:, my_columns.index('c1')] = tmp_sim_data[2]['c1']
    tmp_data.iloc[:, my_columns.index('c2')] = tmp_sim_data[2]['c2']
    tmp_data.iloc[:, my_columns.index('kde_type')] = 'gaussian'
    tmp_data.iloc[:, my_columns.index('bandwidth_rule')] = 'silverman'
    tmp_data.iloc[:, my_columns.index('mixture')] = str(mixture_p)
    tmp_data.iloc[:, my_columns.index('model')] = model
    #tmp_data.iloc[:, my_columns.index('fid')] = file_name
    
    # Get simulated data from kde
    tmp_kde = kde_class.logkde(tmp_sim_data)
    tmp_kde_samples = tmp_kde.kde_sample(n_samples = n_kde)

    tmp_data.iloc[:n_kde, my_columns.index('rt')] = tmp_kde_samples[0]
    tmp_data.iloc[:n_kde, my_columns.index('choice')] = tmp_kde_samples[1]
    tmp_data.iloc[:n_kde, my_columns.index('log_l')] = tmp_kde.kde_eval(data = tmp_kde_samples)

    # Generate rest of the data (to make sure we observe outliers that are negative and close to max_rt)
    # Negative uniform part
    choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'], 
                                  size = n_unif_down)
    rt_tmp = np.random.uniform(low = -1, 
                               high = 0.0001, 
                               size = n_unif_down)

    tmp_data.iloc[n_kde:(n_kde + n_unif_down), my_columns.index('rt')] = rt_tmp
    tmp_data.iloc[n_kde:(n_kde + n_unif_down), my_columns.index('choice')] = choice_tmp
    tmp_data.iloc[n_kde:(n_kde + n_unif_down), my_columns.index('log_l')] = -66.77497 # the number corresponds to log(1e-29)

    # Positive uniform part
    choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'], 
                                  size = n_unif_up)
    rt_tmp = np.random.uniform(low = 0.0001, 
                               high = tmp_sim_data[2]['max_t'], 
                               size = n_unif_up)

    tmp_data.iloc[(n_kde + n_unif_down): , my_columns.index('rt')] = rt_tmp
    tmp_data.iloc[(n_kde + n_unif_down): , my_columns.index('choice')] = choice_tmp
    tmp_data.iloc[(n_kde + n_unif_down):, my_columns.index('log_l')] = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp))
    
    # Some code to get the exact right files name
    # TO BE FIXED : (uuid needs to be separated from 'seconds' component of file name in the 'base_simulations' folder
    
    # Write to file
    if target_file_format == 'csv':
        tmp_data.to_csv(to_folder + '/kde_' + model + '_' + file_name[-39:-7] + '.csv', index = False)
    if target_file_format == 'pickle':
        tmp_data.to_pickle(to_folder + '/kde_' + model + '_' + file_name[-39:-7] + '.pickle')
    
    # Print statements
    print('Finished ' + file_name)
    print('Files generated at this point: ', len(os.listdir(to_folder)))

    
def kde_shuffle_data(from_folder = '',
                     to_folder = '', # note: if we skip copying, the to folder
                     file_chunk_size = 1000,
                     sort_rounds = 20,
                     n_files_total  = 'all',
                     print_info = True,
                     skip_copy_step = True,
                     skip_shuffle_step = False):
    
    # Step 1: Copy to target folder and rename to 'batch_#'
    if not skip_copy_step:
        if n_files_total == 'all':
            file_list = os.listdir(from_folder)    
        else:
            file_list = os.listdir(from_folder)[:n_files_total]
        
        file_cnt = 1
        for tmp_file in file_list:
            pd.read_pickle(from_folder + '/' + tmp_file).to_pickle(to_folder + '/batch_' + str(file_cnt) + '.pickle')
            if (file_cnt % 100) == 0:
               print('files copied: ', file_cnt)
            file_cnt +=1

        print('finished copying files to: ', to_folder)
    
    # Step 2: SHUFFLING
    if not skip_shuffle_step:
        # split file_list into chunks of length: file_chunk_size
        batch_list = os.listdir(to_folder)
        batch_len = len(batch_list)
        chunks = []

        remaining_files = batch_len
        while remaining_files > 0:
            if (remaining_files - file_chunk_size) >= 0:
                chunks.append(file_chunk_size)
            else:
                chunks.append(remaining_files)
            remaining_files = remaining_files - file_chunk_size
        #return(chunks)

        for sort_rnd in range(0, sort_rounds, 1):
            print('Sorting round: ', sort_rnd)

            # Initialize some counters
            file_idx_min = 0
            chunk_cnt = 0

            # Shuffle file list
            batch_list = np.random.choice(batch_list, size = batch_len, replace = False)

            for chunk in chunks:
                print('Chunk ', chunk_cnt, ' of ', len(chunks))
                #print(batch_list)
                chunk_cnt += 1

                # List files in current chunk of files
                tmp_files = batch_list[file_idx_min:(file_idx_min + chunk)]
                tmp_pd_list = []
                file_idx_min += chunk

                # Read files into a list of panda dataframes
                for tmp_file in tmp_files:
                    tmp_pd_list.append(pd.read_pickle(to_folder + '/' + tmp_file))
                    os.remove(to_folder + '/' + tmp_file)

                # Concatenate 
                chunk_pd = pd.concat(tmp_pd_list)
                # Shuffle
                chunk_pd = chunk_pd.sample(frac = 1).reset_index(drop = True)
                # Add grouping variable
                chunk_pd['group'] = np.repeat([i for i in range(0, chunk, 1)], repeats = tmp_pd_list[0].shape[0])
                # Group
                chunk_grouped_pd = chunk_pd.groupby('group') 
                # Split and write to files
                for tmp_group in range(0, chunk, 1):
                    chunk_grouped_pd.get_group(tmp_group).drop(['group'], axis = 1).reset_index(drop = True).to_pickle(to_folder + '/' + tmp_files[tmp_group], protocol = 4)
                    
                    
def kde_load_data(folder = ''):
    # Load training data from file
    train_features = pd.read_pickle(folder + '/train_features.pickle')
    train_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/train_labels.pickle')))
    
    # Load test data from file
    test_features = pd.read_pickle(folder + '/test_features.pickle')
    test_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/test_labels.pickle')))
                               
    # Preprocess labels 
    # 1. get rid of numbers smaller than log(1e-29)
    # 2. Take exponent 
    train_labels[train_labels < np.log(1e-29)] = np.log(1e-29)
    train_labels = np.exp(train_labels)
                                
    test_labels[test_labels < np.log(1e-29)] = np.log(1e-29)
    test_labels = np.exp(test_labels)
                               
    return train_features, train_labels, test_features, test_labels