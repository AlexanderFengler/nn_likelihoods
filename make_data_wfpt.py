import numpy as np
import scipy as scp
import pandas as pd
from datetime import datetime

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

# Generate training / test data for DDM
# We want training data for
# v ~ U(-3,3)
# a ~ U[0.1, 3]
# w ~ U(0,1)
# rt ~ random.sample({-1, 1}) * GAMMA(scale = 1, shape = 2)

def gen_ddm_features(v_range = [-3, 3], a_range = [0.1, 3], w_range = [0, 1], rt_params = [1,2] ,n_samples = 20000):
    data = pd.DataFrame(np.zeros((n_samples, 4)), columns = ['v', 'a', 'w', 'rt'])
    for i in np.arange(0, n_samples, 1):
        data.iloc[i] = [np.random.uniform(low = v_range[0], high = v_range[1], size = 1),
                        np.random.uniform(low = a_range[0], high = a_range[1], size = 1),
                        np.random.uniform(low = w_range[0], high = w_range[1], size = 1),
                        np.random.choice([-1,1], size = 1) * np.random.gamma(rt_params[0], rt_params[1], size = 1)]
    return data

def gen_ddm_labels(data = [1,1,0,1], eps = 10**(-29)):
    labels = np.zeros((data.shape[0],1))
    #labels = pd.Series(np.zeros((data.shape[0],)), name = 'nf_likelihood')
    for i in np.arange(0, labels.shape[0], 1):
        labels[i] = fptd(t = data.loc[i, 'rt'],
                            v = data.loc[i, 'v'],
                            a = data.loc[i, 'a'],
                            w = data.loc[i, 'w'],
                            eps = eps)
    return labels

def make_data(v_range = [-3, 3],
                 a_range = [0.1, 3],
                 w_range = [0, 1],
                 rt_params = [1,2],
                 n_samples = 20000,
                 eps = 10**(-29)):

    data_features = gen_ddm_features(v_range = v_range,
                                    a_range = a_range,
                                    w_range = w_range,
                                    rt_params = rt_params,
                                    n_samples = n_samples)

    data_labels = gen_ddm_labels(data = data_features,
                         eps = eps)

    data_features['nf_likelihood'] = data_labels

    return data_features.copy()


def train_test_split(data = [],
                     p_train = 0.8,
                     write_to_file = False):

    train_indices = np.random.choice([0,1], size = data.shape[0], p = [p_train, 1 - p_train])

    train = data.loc[train_indices == 0].copy()
    test = data.loc[train_indices == 1].copy()

    train_labels = np.asmatrix(train['nf_likelihood'].copy()).T
    train_features = train.drop(labels = 'nf_likelihood', axis = 1).copy()

    test_labels = np.asmatrix(test['nf_likelihood'].copy()).T
    test_features = test.drop(labels = 'nf_likelihood', axis = 1).copy()

    if write_to_file == True:
        cur_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        train.to_csv('train_data_' + cur_time + '.csv')
        test.to_csv('test_data_' + cur_time + '.csv')


    return (train_features.to_dict(orient = 'list'),
            train_labels,
            test_features.to_dict(orient = 'list'),
            test_labels)
