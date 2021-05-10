import pandas as pd
import numpy as np
#import re
import argparse
import sys
import pickle
from cddm_data_simulation import ddm 
from cddm_data_simulation import ddm_flexbound
from cddm_data_simulation import levy_flexbound
from cddm_data_simulation import ornstein_uhlenbeck
from cddm_data_simulation import full_ddm
from cddm_data_simulation import ddm_sdv
#from cddm_data_simulation import ddm_flexbound_pre
from cddm_data_simulation import race_model
from cddm_data_simulation import lca
from cddm_data_simulation import ddm_flexbound_seq2
from cddm_data_simulation import ddm_flexbound_par2
from cddm_data_simulation import ddm_flexbound_mic2

import cddm_data_simulation as cds
import boundary_functions as bf

def bin_simulator_output_pointwise(out = [0, 0],
                                   bin_dt = 0.04,
                                   nbins = 0): # ['v', 'a', 'w', 'ndt', 'angle']
    out_copy = deepcopy(out)

    # Generate bins
    if nbins == 0:
        nbins = int(out[2]['max_t'] / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
        bins[nbins] = np.inf
    else:  
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, len(out[2]['possible_choices']) ) )
    
    #data_out = pd.DataFrame(np.zeros(( columns = ['rt', 'response'])
    out_copy_tmp = deepcopy(out_copy)
    for i in range(out_copy[0].shape[0]):
        for j in range(1, bins.shape[0], 1):
            if out_copy[0][i] > bins[j - 1] and out_copy[0][i] < bins[j]:
                out_copy_tmp[0][i] = j - 1
    out_copy = out_copy_tmp
    #np.array(out_copy[0] / (bins[1] - bins[0])).astype(np.int32)
    
    out_copy[1][out_copy[1] == -1] = 0
    
    return np.concatenate([out_copy[0], out_copy[1]], axis = -1).astype(np.int32)

def bin_simulator_output(out = None,
                         bin_dt = 0.04,
                         nbins = 0,
                         max_t = -1,
                         freq_cnt = False):    # ['v', 'a', 'w', 'ndt', 'angle']

    if max_t == -1:
        max_t = out[2]['max_t']
    
    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:  
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, len(out[2]['possible_choices']) ) )

    for choice in out[2]['possible_choices']:
        counts[:, cnt] = np.histogram(out[0][out[1] == choice], bins = bins)[0]
        cnt += 1

    if freq_cnt == False:
        counts = counts / out[2]['n_samples']
        
    return counts

def bin_arbitrary_fptd(out = None,
                       bin_dt = 0.04,
                       nbins = 256,
                       nchoices = 2,
                       choice_codes = [-1.0, 1.0],
                       max_t = 10.0): # ['v', 'a', 'w', 'ndt', 'angle']

    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:    
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, nchoices) ) 

    for choice in choice_codes:
        counts[:, cnt] = np.histogram(out[:, 0][out[:, 1] == choice], bins = bins)[0] 
        print(np.histogram(out[:, 0][out[:, 1] == choice], bins = bins)[1])
        cnt += 1
    return counts



def simulator(theta, 
              model = 'angle', 
              n_samples = 1000,
              n_trials = 1,
              delta_t = 0.001,
              max_t = 20,
              cartoon = False,
              bin_dim = None,
              bin_pointwise = False): 
    
    # Useful for sbi
    if type(theta) == list:
        print('theta is supplied as list --> simulator assumes n_trials = 1')
        theta = np.asarray(theta).astype(np.float32)
    elif type(theta) == np.ndarray:
        theta = theta.astype(np.float32)
    else:
        theta = theta.numpy()
    
    if len(theta.shape) < 2:
        theta = np.expand_dims(theta, axis = 0)
    if theta.shape[0] != n_trials:
        print('ERROR number of trials does not match first dimension of theta array')
        return

    # 2 choice models 
    if cartoon:
        s = 0.0
    else: 
        s = 1.0

    if model == 'test':
        x = ddm_flexbound(v = theta[:, 0],
                          a = theta[:, 1], 
                          w = theta[:, 2],
                          ndt = theta[:, 3],
                          s = s,
                          n_samples = n_samples,
                          n_trials = n_trials,
                          delta_t = delta_t,
                          boundary_params = {},
                          boundary_fun = bf.constant,
                          boundary_multiplicative = True,
                          max_t = max_t)
    
    if model == 'ddm' or model == 'ddm_elife' or model == 'ddm_analytic':
        x = ddm_flexbound(v = theta[:, 0],
                          a = theta[:, 1], 
                          w = theta[:, 2],
                          ndt = theta[:, 3],
                          s = s,
                          n_samples = n_samples,
                          n_trials = 1,
                          delta_t = delta_t,
                          boundary_params = {},
                          boundary_fun = bf.constant,
                          boundary_multiplicative = True,
                          max_t = max_t)
    
    if model == 'angle' or model == 'angle2':
        x = ddm_flexbound(v = theta[:, 0], 
                          a = theta[:, 1],
                          w = theta[:, 2], 
                          ndt = theta[:, 3], 
                          s = s,
                          boundary_fun = bf.angle, 
                          boundary_multiplicative = False,
                          boundary_params = {'theta': theta[:, 4]}, 
                          delta_t = delta_t,
                          n_samples = n_samples,
                          n_trials = n_trials,
                          max_t = max_t)
    
    if model == 'weibull_cdf' or model == 'weibull_cdf2' or model == 'weibull_cdf_ext' or model == 'weibull_cdf_concave' or model == 'weibull':
        x = ddm_flexbound(v = theta[:, 0], 
                          a = theta[:, 1], 
                          w = theta[:, 2], 
                          ndt = theta[:, 3], 
                          s = s,
                          boundary_fun = bf.weibull_cdf, 
                          boundary_multiplicative = True, 
                          boundary_params = {'alpha': theta[:, 4], 'beta': theta[:, 5]}, 
                          delta_t = delta_t,
                          n_samples = n_samples,
                          n_trials = n_trials,
                          max_t = max_t)
    
    if model == 'levy':
        x = levy_flexbound(v = theta[:, 0], 
                           a = theta[:, 1], 
                           w = theta[:, 2], 
                           alpha_diff = theta[:, 3], 
                           ndt = theta[:, 4],
                           s = s, 
                           boundary_fun = bf.constant, 
                           boundary_multiplicative = True, 
                           boundary_params = {},
                           delta_t = delta_t,
                           n_samples = n_samples,
                           n_trials = n_trials,
                           max_t = max_t)
    
    if model == 'full_ddm' or model == 'full_ddm2':
        x = full_ddm(v = theta[:, 0],
                     a = theta[:, 1],
                     w = theta[:, 2], 
                     ndt = theta[:, 3], 
                     dw = theta[:, 4], 
                     sdv = theta[:, 5], 
                     dndt = theta[:, 6], 
                     s = s,
                     boundary_fun = bf.constant, 
                     boundary_multiplicative = True, 
                     boundary_params = {}, 
                     delta_t = delta_t,
                     n_samples = n_samples,
                     n_trials = n_trials,
                     max_t = max_t)

    if model == 'ddm_sdv':
        x = ddm_sdv(v = theta[:, 0], 
                    a = theta[:, 1], 
                    w = theta[:, 2], 
                    ndt = theta[:, 3],
                    sdv = theta[:, 4],
                    s = s,
                    boundary_fun = bf.constant,
                    boundary_multiplicative = True, 
                    boundary_params = {},
                    delta_t = delta_t,
                    n_samples = n_samples,
                    n_trials = n_trials,
                    max_t = max_t)
        
    if model == 'ornstein' or model == 'ornstein_uhlenbeck':
        x = ornstein_uhlenbeck(v = theta[:, 0], 
                               a = theta[:, 1], 
                               w = theta[:, 2], 
                               g = theta[:, 3], 
                               ndt = theta[:, 4],
                               s = s,
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {},
                               delta_t = delta_t,
                               n_samples = n_samples,
                               n_trials = n_trials,
                               max_t = max_t)

    # 3 Choice models
    if cartoon:
        s = np.tile(np.array([0.0, 0.0, 0.0], dtype = np.float32), (n_trials, 1))
    else:
        s = np.tile(np.array([1.0, 1.0, 1.0], dtype = np.float32), (n_trials, 1))

    if model == 'race_model_3' or model == 'race_3':
        x = race_model(v = theta[:, :3],
                       a = theta[:, [3]],
                       w = theta[:, 4:7],
                       ndt = theta[:, [7]],
                       s = s,
                       boundary_fun = bf.constant,
                       boundary_multiplicative = True,
                       boundary_params = {},
                       delta_t = delta_t,
                       n_samples = n_samples,
                       n_trials = n_trials,
                       max_t = max_t)
        
    if model == 'lca_3':
        x = lca(v = theta[:, :3],
                a = theta[:, [4]],
                w = theta[:, 4:7],
                g = theta[:, [7]],
                b = theta[:, [8]],
                ndt = theta[:, [9]],
                s = s,
                boundary_fun = bf.constant,
                boundary_multiplicative = True,
                boundary_params = {},
                delta_t = delta_t,
                n_samples = n_samples,
                n_trials = n_trials,
                max_t = max_t)


    # 4 Choice models
    if cartoon:
        s = np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype = np.float32), (n_trials, 1))
    else:
        s = np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32), (n_trials, 1))

    if model == 'race_model_4' or model == 'race_4':
        x = race_model(v = theta[:, :4],
                       a = theta[:, [4]],
                       w = theta[:, 5:9],
                       ndt = theta[:, [9]],
                       s = s,
                       boundary_fun = bf.constant,
                       boundary_multiplicative = True,
                       boundary_params = {},
                       delta_t = delta_t,
                       n_samples = n_samples,
                       n_trials = n_trials,
                       max_t = max_t)
        
    if model == 'lca_4':
        x = lca(v = theta[:, :4],
                a = theta[:, [4]],
                w = theta[:, 5:9],
                g = theta[:, [9]],
                b = theta[:, [10]],
                ndt = theta[:, [11]],
                s = s,
                boundary_fun = bf.constant,
                boundary_multiplicative = True,
                boundary_params = {},
                delta_t = delta_t,
                n_samples = n_samples,
                n_trials = n_trials,
                max_t = max_t)

    # Seq / Parallel models (4 choice)
    if cartoon:
        s = 0.0
    else: 
        s = 1.0
        
    if model == 'ddm_seq2':
        x = ddm_flexbound_seq2(v_h = theta[:, 0],
                               v_l_1 = theta[:, 1],
                               v_l_2 = theta[:, 2],
                               a = theta[:, 3],
                               w_h = theta[:, 4],
                               w_l_1 = theta[:, 5],
                               w_l_2 = theta[:, 6],
                               ndt = theta[:, 7],
                               s = s,
                               n_samples = n_samples,
                               n_trials = n_trials,
                               delta_t = delta_t,
                               max_t = max_t,
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {})

    if model == 'ddm_par2':
        x = ddm_flexbound_par2(v_h = theta[:, 0],
                               v_l_1 = theta[:, 1],
                               v_l_2 = theta[:, 2],
                               a = theta[:, 3],
                               w_h = theta[:, 4],
                               w_l_1 = theta[:, 5],
                               w_l_2 = theta[:, 6],
                               ndt = theta[:, 7],
                               s = s,
                               n_samples = n_samples,
                               n_trials = n_trials,
                               delta_t = delta_t,
                               max_t = max_t,
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {})

    if model == 'ddm_mic2':
        x = ddm_flexbound_mic2(v_h = theta[:, 0],
                               v_l_1 = theta[:, 1],
                               v_l_2 = theta[:, 2],
                               a = theta[:, 3],
                               w_h = theta[:, 4],
                               w_l_1 = theta[:, 5],
                               w_l_2 = theta[:, 6],
                               d = theta[:, 7],
                               ndt = theta[:, 8],
                               s = s,
                               n_samples = n_samples,
                               n_trials = n_trials,
                               delta_t = delta_t,
                               max_t = max_t,
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {})
    
    if n_trials == 1:
        #print('passing through')
        #print(x)
        x = (np.squeeze(x[0], axis = 1), np.squeeze(x[1], axis = 1), x[2])

    if bin_dim == 0 or bin_dim == None:
        return x
    elif bin_dim > 0 and not bin_pointwise and n_trials == 1:
        binned_out = bin_simulator_output(x, nbins = bin_dim)
        return (binned_out, x[2])
    elif bin_dim > 0 and bin_pointwise and n_trials == 1:
        binned_out = bin_simulator_output_pointwise(x, nbins = bin_dim)
        return (np.expand_dims(binned_out[:,0], axis = 1), np.expand_dims(binned_out[:, 1], axis = 1), x[2])
    elif bin_dim > 0 and n_trials > 1:
        return 'currently binned outputs not implemented for multi-trial simulators'
    elif bin_dim == -1:
        return 'invalid bin_dim'
    