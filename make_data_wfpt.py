import numpy as np
import scipy as scp
import pandas as pd
from datetime import datetime
import glob
import ddm_data_simulation as ddm_data_simulator
import scipy.integrate as integrate


# WFPT NAVARROS FUSS
def fptd_large(t, w, k):
    terms = np.arange(1, k+1, 1)
    fptd_sum = 0

    for i in terms:
        fptd_sum += i * np.exp( - ((i**2) * (np.pi**2) * t) / 2) * np.sin(i * np.pi * w)
    return fptd_sum * np.pi

def fptd_small(t, w, k):
    temp = (k-1)/2
    flr = np.floor(temp).astype(int)
    cei = - np.ceil(temp).astype(int)
    terms = np.arange(cei, flr + 1, 1)
    #print(terms)
    fptd_sum = 0

    for i in terms:
        fptd_sum += (w + 2 * i) * np.exp( - ((w + 2 * i)**2) / (2 * t))
    return fptd_sum * (1 / np.sqrt(2 * np.pi * (t**3)))

def calculate_leading_term(t, v, a ,w):
    return 1 / (a**2) * np.exp(- (v * a * w) - (((v**2) * t) / 2))

def choice_function(t, eps):
    eps_l = min(eps, 1 / (t * np.pi))
    eps_l = eps
    eps_s = min(eps, 1 / (2 * np.sqrt(2 * np.pi * t)))

    k_l = max(np.sqrt(- (2 * np.log(np.pi * t * eps_l))/(np.pi**2 * t)), 1 / (np.pi * np.sqrt(t)))
    k_s = max(2 + np.sqrt(- 2 * t * np.log(2 * eps_s * np.sqrt(2 * np.pi * t))), 1 + np.sqrt(t))
    return k_s - k_l, k_l, k_s

def fptd(t, v, a, w, eps):
    # negative reaction times signify upper boundary crossing
    # we have to change the parameters as suggested by navarro & fuss (2009)
    if t < 0:
       v = - v
       w = 1 - w
       t = np.abs(t)

    #print('lambda: ' + str(sgn_lambda))
    #print('k_s: ' + str(k_s))
    if t != 0:
        sgn_lambda, k_l, k_s = choice_function(t, eps)
        leading_term = calculate_leading_term(t, v, a, w)
        if sgn_lambda >= 0:
            return max(1e-29, leading_term * fptd_large(t/(a**2), w, k_l))
        else:
            return max(1e-29, leading_term * fptd_small(t/(a**2), w, k_s))
    else:
        return 1e-29

def choice_probabilities(v, a , w):
    return integrate.quad(fptd, 0, 100, args = (v, a, w, 1e-29))[0]

# Generate training / test data for DDM
# We want training data for
# v ~ U(-3,3)
# a ~ U[0.1, 3]
# w ~ U(0,1)
# rt ~ random.sample({-1, 1}) * GAMMA(scale = 1, shape = 2)

def gen_ddm_features_random(v_range = [-3, 3],
                            a_range = [0.1, 3],
                            w_range = [0, 1],
                            rt_params = [1, 2],
                            n_samples = 20000,
                            print_detailed_cnt = False):

    data = pd.DataFrame(np.zeros((n_samples, 5)), columns = ['v', 'a', 'w', 'rt', 'choice'])

    for i in np.arange(0, n_samples, 1):
        data.iloc[i] = [np.random.uniform(low = v_range[0], high = v_range[1], size = 1),
                        np.random.uniform(low = a_range[0], high = a_range[1], size = 1),
                        np.random.uniform(low = w_range[0], high = w_range[1], size = 1),
                        np.random.gamma(rt_params[0], rt_params[1], size = 1),
                        np.random.choice([-1,1], size = 1)]

        if print_detailed_cnt:
            print(str(i))

        if (i % 1000) == 0:
            print('datapoint ' + str(i) + ' generated')
    return data

def gen_ddm_features_sim(v_range = [-3, 3],
                         a_range = [0.1, 3],
                         w_range = [0, 1],
                         n_samples = 20000,
                         print_detailed_cnt = False):

    data = pd.DataFrame(np.zeros((n_samples, 5)), columns = ['v', 'a', 'w', 'rt', 'choice'])

    for i in np.arange(0, n_samples, 1):
        v_tmp = np.random.uniform(low = v_range[0], high = v_range[1], size = 1)
        a_tmp = np.random.uniform(low = a_range[0], high = a_range[1], size = 1)
        w_tmp = np.random.uniform(low = w_range[0], high = w_range[1], size = 1)
        rt_tmp, choice_tmp = ddm_data_simulator.ddm_simulate(v = v_tmp,
                                                             a = a_tmp,
                                                             w = w_tmp,
                                                             n_samples = 1,
                                                             print_info = False
                                                             )
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

def gen_ddm_labels(data = [1,1,0,1], eps = 10**(-29)):
    labels = np.zeros((data.shape[0],1))
    #labels = pd.Series(np.zeros((data.shape[0],)), name = 'nf_likelihood')
    for i in np.arange(0, labels.shape[0], 1):
        labels[i] = fptd(t = data.loc[i, 'rt'] * data.loc[i, 'choice'],
                            v = data.loc[i, 'v'],
                            a = data.loc[i, 'a'],
                            w = data.loc[i, 'w'],
                            eps = eps)
        if (i % 1000) == 0:
            print('label ' + str(i) + ' generated')

    return labels

def make_data_rt_choice(v_range = [-3, 3],
                        a_range = [0.1, 3],
                        w_range = [0, 1],
                        rt_params = [1,2],
                        n_samples = 20000,
                        eps = 10**(-29),
                        f_signature = '',
                        write_to_file = True,
                        method = 'random',
                        print_detailed_cnt = False):

    if method == 'random':
        data_features = gen_ddm_features_random(v_range = v_range,
                                                a_range = a_range,
                                                w_range = w_range,
                                                rt_params = rt_params,
                                                n_samples = n_samples,
                                                print_detailed_cnt = print_detailed_cnt)

    if method == 'sim':
        data_features = gen_ddm_features_sim(v_range = v_range,
                                             a_range = a_range,
                                             w_range = w_range,
                                             n_samples = n_samples,
                                             print_detailed_cnt = print_detailed_cnt)

    data_labels = pd.DataFrame(gen_ddm_labels(data = data_features,
                               eps = eps),
                               columns = ['nf_likelihood']
                               )

    data = pd.concat([data_features, data_labels], axis = 1)

    cur_time = datetime.now().strftime('%m_%d_%y_%H_%M_%S')

    if write_to_file == True:
       data.to_csv('data_storage/data_' + str(n_samples) + '_' + f_signature + cur_time + '.csv')

    return data.copy(), cur_time, n_samples

def make_data_choice_probabilities(v_range = [-3, 3],
                                   a_range = [0.1, 3],
                                   w_range = [0, 1],
                                   eps = 1e-29,
                                   f_signature = '',
                                   write_to_file = True,
                                   print_detailed_cnt = False):


    data = pd.DataFrame(np.zeros((n_samples, 5)), columns = ['v',
                                                             'a',
                                                             'w',
                                                             'p_lower_barrier'])

    for i in np.arange(0, n_samples, 1):
        v_tmp = np.random.uniform(low = v_range[0], high = v_range[1], size = 1)
        a_tmp = np.random.uniform(low = a_range[0], high = a_range[1], size = 1)
        w_tmp = np.random.uniform(low = w_range[0], high = w_range[1], size = 1)
        p_tmp = choice_probabilities(v = v_tmp,
                                   a = a_tmp,
                                   w = w_tmp
                                   )

        data.iloc[i] = [v_tmp,
                      a_tmp,
                      w_tmp,
                      p_tmp
                      ]

        if print_detailed_cnt:
          print(str(i))

        if (i % 1000) == 0:
          print('datapoint ' + str(i) + ' generated')

    return data

def train_test_split_choice_probabilities(data = [],
                                          p_train = 0.8,
                                          write_to_file = True,
                                          from_file = True,
                                          f_signature = '',  # default behavior is to load the latest file of a specified number of examples
                                          n = None): # if we pass a number, we pick a data file with the specified number of examples, if None the function picks some data file

    assert n != None, 'please specify the size of the dataset (rows) that is supposed to be read in....'

    if from_file:

        # List data files in directory
        if f_signature == '':
            flist = glob.glob('data_storage/data_' + str(n) + '*')
            assert len(flist) > 0, 'There seems to be no datafile that fullfills the requirements passed to the function'
            fname = flist[-1]
            data = pd.read_csv(fname)
        else:
            list = glob.glob('data_storage/data_' + str(n) + f_signature + '*')
            assert len(flist) > 0, 'There seems to be no datafile that fullfills the requirements passed to the function'
            fname = flist[-1]
            data = pd.read_csv(fname)
            data = pd.read_csv('data_storage/data_' + str(n) + f_signature + '*')

    n = data.shape[0]
    train_indices = np.random.choice([0,1], size = data.shape[0], p = [p_train, 1 - p_train])

    train = data.loc[train_indices == 0].copy()
    test = data.loc[train_indices == 1].copy()

    train_labels = np.asmatrix(train['p_lower_barrier'].copy()).T
    train_features = train.drop(labels = 'p_lower_barrier', axis = 1).copy()

    test_labels = np.asmatrix(test['p_lower_barrier'].copy()).T
    test_features = test.drop(labels = 'p_lower_barrier', axis = 1).copy()

    if write_to_file == True:
        print('writing training and test data to file ....')
        train.to_csv('data_storage/train_data_' + str(n) + f_signature + fname[-21:])
        test.to_csv('data_storage/test_data_' + str(n) + f_signature + fname[-21:])
        np.savetxt('data_storage/train_indices_' + str(n) + f_signature + fname[-21:], train_indices, delimiter = ',')


    # clean up dictionary: Get rid of index coltrain_features = train_features[['v', 'a', 'w', 'rt', 'choice']], which is unfortunately retained when reading with 'from_csv'
    train_features = train_features[['v', 'a', 'w', 'p_lower_barrier']]
    test_features = test_features[['v', 'a', 'w', 'p_lower_barrier']]

    # Transform feature pandas into dicts as expected by tensorflow
    train_features = train_features.to_dict(orient = 'list')
    test_features = test_features.to_dict(orient = 'list')

    return (train_features,
            train_labels,
            test_features,
            test_labels)


def train_test_split_rt_choice(data = [],
                               p_train = 0.8,
                               write_to_file = True,
                               from_file = True,
                               f_signature = '',  # default behavior is to load the latest file of a specified number of examples
                               n = None): # if we pass a number, we pick a data file with the specified number of examples, if None the function picks some data file

    assert n != None, 'please specify the size of the dataset (rows) that is supposed to be read in....'

    if from_file:

        # List data files in directory
        if f_signature == '':
            flist = glob.glob('data_storage/data_' + str(n) + '*')
            assert len(flist) > 0, 'There seems to be no datafile that fullfills the requirements passed to the function'
            fname = flist[-1]
            data = pd.read_csv(fname)
        else:
            list = glob.glob('data_storage/data_' + str(n) + f_signature + '*')
            assert len(flist) > 0, 'There seems to be no datafile that fullfills the requirements passed to the function'
            fname = flist[-1]
            data = pd.read_csv(fname)
            data = pd.read_csv('data_storage/data_' + str(n) + f_signature + '*')

    n = data.shape[0]
    train_indices = np.random.choice([0,1], size = data.shape[0], p = [p_train, 1 - p_train])

    train = data.loc[train_indices == 0].copy()
    test = data.loc[train_indices == 1].copy()

    train_labels = np.asmatrix(train['nf_likelihood'].copy()).T
    train_features = train.drop(labels = 'nf_likelihood', axis = 1).copy()

    test_labels = np.asmatrix(test['nf_likelihood'].copy()).T
    test_features = test.drop(labels = 'nf_likelihood', axis = 1).copy()

    if write_to_file == True:
        print('writing training and test data to file ....')
        train.to_csv('data_storage/train_data_' + str(n) + '_' + f_signature + fname[-21:])
        test.to_csv('data_storage/test_data_' + str(n) + '_' + f_signature + fname[-21:])
        np.savetxt('data_storage/train_indices_' + str(n) + '_' + f_signature + fname[-21:], train_indices, delimiter = ',')


    # clean up dictionary: Get rid of index coltrain_features = train_features[['v', 'a', 'w', 'rt', 'choice']], which is unfortunately retained when reading with 'from_csv'
    train_features = train_features[['v', 'a', 'w', 'rt', 'choice']]
    test_features = test_features[['v', 'a', 'w', 'rt', 'choice']]

    # Transform feature pandas into dicts as expected by tensorflow
    train_features = train_features.to_dict(orient = 'list')
    test_features = test_features.to_dict(orient = 'list')

    return (train_features,
            train_labels,
            test_features,
            test_labels)

def train_test_from_file_rt_choice(
                                   f_signature = '', # default behavior is to load the latest file of a specified number of examples
                                   n = None # if we pass a number, we pick a data file with the specified number of examples, if None the function picks some data file
                                   ):

    assert n != None, 'please specify the size of the dataset (rows) that is supposed to be read in....'

    # List data files in directory
    if f_signature == '':
        flist_train = glob.glob('data_storage/train_data_' + str(n) + '*')
        flist_test = glob.glob('data_storage/test_data_' + str(n) + '*')
        assert len(flist_train) > 0, 'There seems to be no datafile for train data that fullfills the requirements passed to the function'
        assert len(flist_test) > 0, 'There seems to be no datafile for train data that fullfills the requirements passed to the function'
        fname_train = flist_train[-1]
        fname_test = flist_test[-1]

    else:
        flist_train = glob.glob('data_storage/train_data_' + str(n) + f_signature + '*')
        flist_test = glob.glob('data_storage/test_data_' + str(n) + f_signature + '*')
        assert len(flist_train) > 0, 'There seems to be no datafile for train data that fullfills the requirements passed to the function'
        assert len(flist_test) > 0, 'There seems to be no datafile for train data that fullfills the requirements passed to the function'
        fname_train = flist_train[-1]
        fname_test =  flist_test[-1]


    print('datafile used to read in training data: ' + flist_train[-1])
    print('datafile used to read in test data: ' + flist_test[-1])

    # Reading in the data
    train_data = pd.read_csv(fname_train)
    test_data = pd.read_csv(fname_test)

    # Splitting into labels and features
    train_labels = np.asmatrix(train_data['nf_likelihood'].copy()).T
    train_features = train_data.drop(labels = 'nf_likelihood', axis = 1).copy()

    test_labels = np.asmatrix(test_data['nf_likelihood'].copy()).T
    test_features = test_data.drop(labels = 'nf_likelihood', axis = 1).copy()

    # clean up dictionary: Get rid of index coltrain_features = train_features[['v', 'a', 'w', 'rt', 'choice']], which is unfortunately retained when reading with 'from_csv'
    train_features = train_features[['v', 'a', 'w', 'rt', 'choice']]
    test_features = test_features[['v', 'a', 'w', 'rt', 'choice']]

    # Transform feature pandas into dicts as expected by tensorflow
    train_features = train_features.to_dict(orient = 'list')
    test_features = test_features.to_dict(orient = 'list')

    return (train_features,
            train_labels,
            test_features,
            test_labels)


def train_test_from_file_choice_probabilities(
                                              f_signature = '', # default behavior is to load the latest file of a specified number of examples
                                              n = None # if we pass a number, we pick a data file with the specified number of examples, if None the function picks some data file
                                              ):

    assert n != None, 'please specify the size of the dataset (rows) that is supposed to be read in....'

    # List data files in directory
    if f_signature == '':
        flist_train = glob.glob('data_storage/train_data_' + str(n) + '*')
        flist_test = glob.glob('data_storage/test_data_' + str(n) + '*')
        assert len(flist_train) > 0, 'There seems to be no datafile for train data that fullfills the requirements passed to the function'
        assert len(flist_test) > 0, 'There seems to be no datafile for train data that fullfills the requirements passed to the function'
        fname_train = flist_train[-1]
        fname_test = flist_test[-1]

    else:
        flist_train = glob.glob('data_storage/train_data_' + str(n) + f_signature + '*')
        flist_test = glob.glob('data_storage/test_data_' + str(n) + f_signature + '*')
        assert len(flist_train) > 0, 'There seems to be no datafile for train data that fullfills the requirements passed to the function'
        assert len(flist_test) > 0, 'There seems to be no datafile for train data that fullfills the requirements passed to the function'
        fname_train = flist_train[-1]
        fname_test =  flist_test[-1]


    print('datafile used to read in training data: ' + flist_train[-1])
    print('datafile used to read in test data: ' + flist_test[-1])

    # Reading in the data
    train_data = pd.read_csv(fname_train)
    test_data = pd.read_csv(fname_test)

    # Splitting into labels and features
    train_labels = np.asmatrix(train_data['p_lower_barrier'].copy()).T
    train_features = train_data.drop(labels = 'p_lower_barrier', axis = 1).copy()

    test_labels = np.asmatrix(test_data['p_lower_barrier'].copy()).T
    test_features = test_data.drop(labels = 'p_lower_barrier', axis = 1).copy()

    # clean up dictionary: Get rid of index coltrain_features = train_features[['v', 'a', 'w', 'rt', 'choice']], which is unfortunately retained when reading with 'from_csv'
    train_features = train_features[['v', 'a', 'w', 'p_lower_barrier']]
    test_features = test_features[['v', 'a', 'w', 'p_lower_barrier']]

    # Transform feature pandas into dicts as expected by tensorflow
    train_features = train_features.to_dict(orient = 'list')
    test_features = test_features.to_dict(orient = 'list')

    return (train_features,
            train_labels,
            test_features,
            test_labels)
