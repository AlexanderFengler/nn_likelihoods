# Basic python 
import numpy as np
import scipy as scp
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

def kde_train_test_from_simulations_flexbound(base_simulation_folder = '',
                                              target_folder = '',
                                              n_total = 10000,
                                              p_train = 0.8,
                                              mixture_p = [0.8, 0.1, 0.1], # maybe here I can instead pass a function that provides a sampler
                                              process_params = ['v', 'a', 'w', 'c1', 'c2'],
                                              model = 'ddm_flexbound',
                                              print_info = False,
                                              target_file_format = 'pickle',
                                              n_files_max = 'all'):
    
    # Get files 
    #print(base_simulation_folder)
    files_ = os.listdir(base_simulation_folder)
    random.shuffle(files_)
    #print(files_)
    # Compute some quantities
    
    if n_files_max == 'all':
        n_files = len(files_)
    else:
        n_files = n_files_max
        files_ = np.random.choice(files_ , n_files)
    
    files_ = [base_simulation_folder + '/' + file_ for file_ in files_]
    
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
    
    
    # Main while loop
    row_cnt = 0 
    cnt = 0
    for file_ in files_:
        tmp_sim_data = pickle.load( open( file_ , "rb" ) )
        
        for param in process_params:
            data.iloc[row_cnt:(row_cnt + n_samples_by_kde), my_columns.index(param)] = tmp_sim_data[2][param]
            
                        
#             data.iloc[row_cnt:(row_cnt + n_samples_by_kde), my_columns.index('v')] = tmp_sim_data[2]['v']            
#             data.iloc[row_cnt:(row_cnt + n_samples_by_kde), my_columns.index('a')] = tmp_sim_data[2]['a']
#             data.iloc[row_cnt:(row_cnt + n_samples_by_kde), my_columns.index('w')] = tmp_sim_data[2]['w']      
#             data.iloc[row_cnt:(row_cnt + n_samples_by_kde), my_columns.index('c1')] = tmp_sim_data[2]['c1']
#             data.iloc[row_cnt:(row_cnt + n_samples_by_kde), my_columns.index('c2')] = tmp_sim_data[2]['c2']
    
        # Get simulated data from kde
        tmp_kde = kde_class.logkde(tmp_sim_data)
        tmp_kde_samples = tmp_kde.kde_sample(n_samples = n_kde)

        data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('rt')] = tmp_kde_samples[0].ravel()
        data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('choice')] = tmp_kde_samples[1].ravel()
        data.iloc[row_cnt:(row_cnt + n_kde), my_columns.index('log_l')] = tmp_kde.kde_eval(data = tmp_kde_samples).ravel()

        # Generate rest of the data (to make sure we observe outliers that are negative and close to max_rt)
        
        # Negative uniform part
        choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'], 
                                      size = n_unif_down)
        
        rt_tmp = np.random.uniform(low = - 1, 
                                   high = 0.0001, 
                                   size = n_unif_down)

        data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('rt')] = rt_tmp
        data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('choice')] = choice_tmp
        data.iloc[(row_cnt + n_kde):(row_cnt + n_kde + n_unif_down), my_columns.index('log_l')] = -66.77497 # the number corresponds to log(1e-29)

        # Positive uniform part
        choice_tmp = np.random.choice(tmp_sim_data[2]['possible_choices'], 
                                      size = n_unif_up)
        
        rt_tmp = np.random.uniform(low = 0.0001, 
                                   high = tmp_sim_data[2]['max_t'], 
                                   size = n_unif_up)

        data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('rt')] = rt_tmp
        data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('choice')] = choice_tmp
        data.iloc[(row_cnt + n_kde + n_unif_down):(row_cnt + n_samples_by_kde), my_columns.index('log_l')] = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp))
        
        row_cnt += n_samples_by_kde
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt, 'kdes generated')
    
    
    # Shuffle Data Frame
    np.random.shuffle(data.values)
    #data = data.sample(frac = 1).reset_index(drop = True, inplace = True)
    
    
    train_id = np.random.choice(a = [True, False], 
                                size = data.shape[0], 
                                replace = True, 
                                p = [p_train, 1 - p_train])
    test_id = np.invert(train_id)
    
    # Store training and test data to file 
    print('writing to train test data to file')
    data.iloc[train_id, :7].to_pickle(target_folder + '/train_features.pickle' , protocol = 4)
    data.iloc[test_id, :7].to_pickle(target_folder + '/test_features.pickle', protocol = 4)
    data.iloc[train_id, 7].to_pickle(target_folder + '/train_labels.pickle', protocol = 4)
    data.iloc[test_id, 7].to_pickle(target_folder + '/test_labels.pickle', protocol = 4)
                        
    # Write metafile (just need info from one of the files in base_simulation folder
    # Hack for now: Just copy one of the base simulations files over
    pickle.dump(tmp_sim_data,  open(target_folder + '/meta_data.pickle', 'wb') )                
    
    return data

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