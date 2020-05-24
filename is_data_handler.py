import os
import pickle
import numpy as np
import re
from string import ascii_letters
from datetime import datetime
import argparse
import yaml


def collect_datasets_is(folder = [],
                        model = [],
                        ndata = [],
                        nsubsample = []):
    
    n_data_substring = 'N_' + str(ndata)
    
    is_dict = {}
    is_dict['gt'] = []
    is_dict['posterior_samples'] = []
    is_dict['timings'] = []
    is_dict['perplexities'] = []
    is_dict['importance_weights'] = []
    is_dict['effective_sample_size'] = []
    is_dict['means'] = []
    
    files_ = os.listdir(folder)
    
    for file_ in files_:
        if model in file_ and n_data_substring in file_ and 'summary' not in file_:
            tmp = pickle.load(open(folder + file_, 'rb'), encoding = 'latin1')
            sub_idx = np.random.choice(tmp['posterior_samples'].shape[0], n_subsample, replace = False) 
            is_dict['gt'].append(tmp['gt_params'])
            is_dict['posterior_samples'].append(tmp['posterior_samples'][sub_idx, :])
            is_dict['timings'].append(tmp['timeToConvergence'])
            is_dict['perplexities'].append(tmp['norm_perplexity'])
            is_dict['importance_weights'].append(tmp['final_w'][sub_idx])
            is_dict['effective_sample_size'].append(1 / np.sum(np.square(tmp['final_w'])))
#             is_dict['means'].append(np.mean(tmp['posterior_samples'], axis = 0))
            
        print('Processed file: ', file_)
    
    is_dict['gt'] = np.stack(is_dict['gt'])
    is_dict['posterior_samples'] = np.stack(is_dict['posterior_samples'])
    is_dict['timings'] = np.array(is_dict['timings'])
    is_dict['perplexities'] = np.array(is_dict['perplexities'])
    is_dict['importance_weights'] = np.stack(is_dict['importance_weights'])
    is_dict['means'] = np.stack(is_dict['means'])
    
    print('writing to file: ', '/users/afengler/OneDrive/project_nn_likelihoods/cogsci/IS_summary_' + model + \
                     '_' + n_data_substring + '.pickle')
    
    pickle.dump(is_dict, open('/users/afengler/OneDrive/project_nn_likelihoods/cogsci/IS_summary_' + model + \
                     '_' + n_data_substring + '.pickle', 'wb'), protocol = 4)
    
    return is_dict

if __name__ == "__main__":
    # Currently available models = ['weibull', 'race_model_6', 'ornstein', 'full_ddm', 'ddm_seq2', 'ddm_par2', 'ddm_mic2']

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--method",
                     nargs = "*"
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--ndata",
                     nargs = "*"
                     type = int,
                     default = 1024)
    CLI.add_argument("--nsubsample",
                     type = int,
                     default = 10000)
    CLI.add_argument("--isfolder",
                     type = str,
                     default = 'cogsci')
    
    args = CLI.parse_args()
    print(args)

    machine = args.machine
    method = args.method 
    ndata = args.ndata
    nsubsample = args.nsubsample
    isfolder = args.isfolder
    
    if machine == 'home':
        is_sample_folder = '/Users/afengler/OneDrive/project_nn_likelihoods/data/' + isfolder + '/'
    
    if machine == 'ccv':  
        is_sample_folder = '/users/afengler/data/' + isfolder + '/'
    
    if machine == 'x7':
        is_sample_folder = '/users/afengler/data/' + isfolder + '/'
        
    for model in method:
        for n in ndata:
            print('Started processing model: ', model, ' with ndata: ', n)
            collect_datasets_is(folder = is_sample_folder,
                                model = model,
                                ndata = n,
                                nsubsample = nsubsample)