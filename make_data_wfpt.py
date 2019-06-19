import numpy as np
import scipy as scp
import pandas as pd
from datetime import datetime
import glob
import uuid
import ddm_data_simulation as ddm_data_simulator
import scipy.integrate as integrate
import dwiener


# Generate training / test data for DDM
# We want training data for
# v ~ U(-3,3)
# a ~ U[0.1, 3]
# w ~ U(0,1)
# rt ~ random.sample({-1, 1}) * GAMMA(scale = 1, shape = 2)

# Function that generate 'features' (ML parlance, here features are (v, a, w, rt, c))
# ----

def gen_ddm_features_random(v_range = [-3, 3],
                            a_range = [0.1, 3],
                            w_range = [0, 1],
                            rt_params = [1, 2],
                            n_samples = 20000,
                            mixture_p = 0.2,
                            print_detailed_cnt = False):

    data = pd.DataFrame(np.zeros((n_samples, 5)), columns = ['v', 'a', 'w', 'rt', 'choice'])
    mixture_indicator = np.random.choice([0, 1, 2],
                                         p = [mixture_p[0], mixture_p[1], mixture_p[2]],
                                         size = n_samples)

    for i in np.arange(0, n_samples, 1):
        if mixture_indicator[i] == 0:
            data.iloc[i] = [np.random.uniform(low = v_range[0], high = v_range[1], size = 1),
                            np.random.uniform(low = a_range[0], high = a_range[1], size = 1),
                            np.random.uniform(low = w_range[0], high = w_range[1], size = 1),
                            np.random.gamma(rt_params[0], rt_params[1], size = 1),
                            np.random.choice([-1, 1], size = 1)]

        elif mixture_indicator[i] == 1:
            data.iloc[i] = [np.random.uniform(low = v_range[0], high = v_range[1], size = 1),
                            np.random.uniform(low = a_range[0], high = a_range[1], size = 1),
                            np.random.uniform(low = w_range[0], high = w_range[1], size = 1),
                            np.random.normal(loc = 0, scale = 1, size = 1),
                            np.random.choice([-1, 1], size = 1)]

        else:
            data.iloc[i] = [np.random.uniform(low = v_range[0], high = v_range[1], size = 1),
                            np.random.uniform(low = a_range[0], high = a_range[1], size = 1),
                            np.random.uniform(low = w_range[0], high = w_range[1], size = 1),
                            np.random.uniform(low = 5.0, high = 20, size = 1),
                            np.random.choice([-1, 1], size = 1)]

        if print_detailed_cnt:
            print(str(i))

        if (i % 1000) == 0:
            print('datapoint ' + str(i) + ' generated')
    return data

def gen_ddm_features_sim(v_range = [-3, 3],
                         a_range = [0.1, 3],
                         w_range = [0, 1],
                         n_samples = 20000,
                         mixture_p = 0.2,
                         print_detailed_cnt = False):

    data = pd.DataFrame(np.zeros((n_samples, 5)), columns = ['v', 'a', 'w', 'rt', 'choice'])
    mixture_indicator = np.random.choice([0, 1, 2], p = [mixture_p[0], mixture_p[1], mixture_p[2]] , size = n_samples)

    for i in np.arange(0, n_samples, 1):
        v_tmp = np.random.uniform(low = v_range[0], high = v_range[1], size = 1)
        a_tmp = np.random.uniform(low = a_range[0], high = a_range[1], size = 1)
        w_tmp = np.random.uniform(low = w_range[0], high = w_range[1], size = 1)

        if mixture_indicator[i] == 0:
            rt_tmp, choice_tmp = ddm_data_simulator.ddm_simulate(v = v_tmp,
                                                                 a = a_tmp,
                                                                 w = w_tmp,
                                                                 n_samples = 1,
                                                                 print_info = False
                                                                 )
        elif mixture_indicator[i] == 1:
            choice_tmp = np.random.choice([-1, 1], size = 1)
            rt_tmp = np.random.normal(loc = 0, scale = 1, size = 1)

        else:
            choice_tmp = np.random.choice([-1, 1], size = 1)
            rt_tmp = np.random.uniform(low = 5.0, high = 20.0, size = 1)

        data.iloc[i] = [v_tmp,
                        a_tmp,
                        w_tmp,
                        rt_tmp,
                        choice_tmp
                        ]

        if print_detailed_cnt:
            print(str(i))

        if (i % 1000) == 0:
            print('datapoint ' + str(i) + ' generated')
    return  data


def gen_ddm_features_combination(v_range = [1,2],
                                 a_range = [1],
                                 w_range = [0.5],
                                 n_samples = 20000,
                                 ):

    data = pd.DataFrame(np.zeros((n_samples, 5)),
                        columns = ['v', 'a', 'w', 'rt', 'choice'])
    mixture_indicator = np.random.choice([0, 1, 2],
                                          p = [mixture_p[0], mixture_p[1], mixture_p[2]],
                                          size = n_samples)
    n_by_parameter_set = n_samples // (len(v_range) * len(a_range) * len(w_range))

    cnt = 0
    for v_tmp in v_range:
        for a_tmp in a_range:
            for w_tmp in w_range:
                for i in range(0, n_by_parameter_set, 1):
                    if mixture_indicator[i] == 0:
                        rt_tmp, choice_tmp = ddm_data_simulator.ddm_simulate(v = v_tmp,
                                                                             a = a_tmp,
                                                                             w = w_tmp,
                                                                             n_samples = 1,
                                                                             print_info = False
                                                                             )

                    elif mixture_indicator[i] == 1:
                        choice_tmp = np.random.choice([-1, 1], size = 1)
                        rt_tmp = np.random.normal(loc = 0, scale = 1, size = 1)
                    else:
                        choice_tmp = np.random.choice([-1, 1], size = 1)
                        rt_tmp = np.random.uniform(low = 5.0, high = 20.0, size = 1)


                    data.iloc[i] = [v_tmp,
                                    a_tmp,
                                    w_tmp,
                                    rt_tmp,
                                    choice_tmp
                                    ]

                    if print_detailed_cnt:
                        print(str(i))

                    cnt += 1
                    if (cnt % 1000) == 0:
                        print('datapoint ' + str(i) + ' generated')

    return  data
# ----

# Function that generates 'Labels' (ML parlance, here 'label' refers to a navarro-fuss likelihood computed for datapoint of the form (v,a,w,rt,c))
# ----
def gen_ddm_labels(data = [1,1,0,1], eps = 10**(-29)):
    labels = np.zeros((data.shape[0],1))
    for i in np.arange(0, labels.shape[0], 1):
        if data.loc[i, 'rt'] <= 0:
            labels[i] = 0
        else:
            labels[i] = dwiener.fptd(t = data.loc[i, 'rt'] * data.loc[i, 'choice'],
                                     v = data.loc[i, 'v'],
                                     a = data.loc[i, 'a'],
                                     w = data.loc[i, 'w'],
                                     eps = eps)

        if (i % 1000) == 0:
            print('label ' + str(i) + ' generated')
    return labels
# ----

# Functions to generate full datasets
# ----
def make_data_rt_choice(v_range = [-3, 3],
                        a_range = [0.1, 3],
                        w_range = [0, 1],
                        rt_params = [1,2],
                        n_samples = 20000,
                        eps = 10**(-29),
                        target_folder = '',
                        write_to_file = True,
                        method = 'random',
                        mixture_p = [0.8, 0.1, 0.1]):

    # Make target folder if it doesn't exist:
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    # Make features -----
    if method == 'random':
        data_features = gen_ddm_features_random(v_range = v_range,
                                                a_range = a_range,
                                                w_range = w_range,
                                                rt_params = rt_params,
                                                n_samples = n_samples,
                                                print_detailed_cnt = print_detailed_cnt,
                                                mixture_p = mixture_p)

    if method == 'sim':
        data_features = gen_ddm_features_sim(v_range = v_range,
                                             a_range = a_range,
                                             w_range = w_range,
                                             n_samples = n_samples,
                                             print_detailed_cnt = print_detailed_cnt,
                                             mixture_p = mixture_p)

    if method == 'discrete_parameter_combinations':
        data_features = gen_dmm_features_combination(v = v_range,
                                                     a = a_range,
                                                     w = w_range,
                                                     n_samples = n_samples,
                                                     print_detailed_cnt = print_detailed_cnt,
                                                     mixture_p = mixture_p)
    # ----

    # Make labels
    data_labels = pd.DataFrame(gen_ddm_labels(data = data_features, eps = eps),
                               columns = ['nf_likelihood'])


    # Column concat features and labels
    data = pd.concat([data_features, data_labels], axis = 1)

    # Write to file
    if write_to_file == True:
       data.to_pickle(target_folder + '/data_' + str(n_samples) + '_' + uuid.uuid1().hex + '.pickle',
                      protocol = 4)

    return data

def make_data_choice_probabilities(v_range = [-3, 3],
                                   a_range = [0.1, 3],
                                   w_range = [0, 1],
                                   n_samples = 20000,
                                   eps = 1e-29,
                                   write_to_file = True,
                                   target_folder = ''
                                   ):

    # Initialize dataframe
    data = pd.DataFrame(np.zeros((n_samples, 4)), columns = ['v',
                                                             'a',
                                                             'w',
                                                             'p_lower_barrier'])

    for i in np.arange(0, n_samples, 1):
        # Features
        v_tmp = np.random.uniform(low = v_range[0], high = v_range[1], size = 1)
        a_tmp = np.random.uniform(low = a_range[0], high = a_range[1], size = 1)
        w_tmp = np.random.uniform(low = w_range[0], high = w_range[1], size = 1)
        # Labels
        p_lower_tmp = choice_probabilities(v = v_tmp,
                                           a = a_tmp,
                                           w = w_tmp)
        # Store in dataframe
        data.iloc[i] = [v_tmp,
                        a_tmp,
                        w_tmp,
                        p_lower_tmp
                       ]

        if (i % 1000) == 0:
          print('datapoint ' + str(i) + ' generated')

    if write_to_file == True:
       data.to_pickle(target_folder + '/data_' + str(n_samples) + '_' + uuid.uuid1().hex + '.pickle',
                      protocol = 4)

    return data
# ----



# Functions that make and load training and test sets from the basic datafiles generated above --------
def make_train_test_split(source_folder = '',
                          target_folder = '',
                          p_train = 0.8):

    # Deal with target folder
    if target_folder == '':
        target_folder = source_folder
    else:
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)


    # Get files in specified folder
    print('get files in folder')
    files_ = os.listdir(folder)

    # Check if folder currently contains a train-test split
    print('check if we have a train and test sets already')
    for file_ in files_:
        if file[:7] == 'train_f':
            return 'looks like a train test split exists in folder: Please remove before running this function'

    # If no train-test split in folder: collect 'data_*' files
    print('folder clean so proceeding...')
    data_files = []
    for file_ in files_:
        if file_[:5] == 'data_':
            data_files.append(file_)

    # Read in and concatenate files
    print('read, concatenate and shuffle data')
    data = pd.concat([pd.read_pickle(folder + file_) for file_ in data_files_])

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
    data.iloc[train_id, :(len(n_cols) - 1)].to_pickle(folder + 'train_features.pickle',
                                                          protocol = 4)

    data.iloc[test_id, :(len(n_cols) - 1)].to_pickle(folder + 'test_features.pickle',
                                                         protocol = 4)

    data.iloc[train_id, (len(n_cols) - 1)].to_pickle(folder + 'train_labels.pickle',
                                                         protocol = 4)

    data.iloc[test_id, (len(n_cols) - 1)].to_pickle(folder + 'test_labels.pickle',
                                                        protocol = 4)

    return 'success'

def load_data(folder = '',
              log = False,
              cutoff = 1e-29 # 'none' or value
              ):

    # Load training data from file
    train_features = pd.read_pickle(folder + '/train_features.pickle')
    train_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/train_labels.pickle')))

    # Load test data from file
    test_features = pd.read_pickle(folder + '/test_features.pickle')
    test_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/test_labels.pickle')))

    # Preprocess labels
    # 1. get rid of labels smaller than cutoff
    if cutoff != 'none':
        train_labels[train_labels < cutoff] = cutoff
        test_labels[test_labels < cutoff] = cutoff

    # 2. Take exp
    if log:
        train_labels = np.log(train_labels)
        test_labels = np.log(test_labels)

    return train_features, train_labels, test_features, test_labels
# -------------------------------------------------------------------------------------------
