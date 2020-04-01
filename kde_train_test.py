#import ddm_data_simulation as ddm_sim
import scipy as scp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import multiprocessing as mp
import psutil
import pickle
import os
import sys
import argparse

import kde_training_utilities as kde_util
import kde_class as kde

if __name__ == "__main__":
    # Interfact ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'ccv')
    CLI.add_argument('--method',
                     type = str,
                     default = 'ddm')
    CLI.add_argument('--simfolder',
                     type = str,
                     default = 'base_simulations')
    CLI.add_argument('--fileprefix',
                     type = str,
                     default = 'ddm_base_simulations')
    CLI.add_argument('--fileid',
                     type = str,
                     default = 'TEST')
    CLI.add_argument('--outfolder',
                     type = str,
                     default = 'train_test')
    CLI.add_argument('--nbyparam',
                     type = int,
                     default = 1000)
    CLI.add_argument('--mixture',
                     nargs = '*',
                     type = float,
                     default = [0.8, 0.1, 0.1])
    CLI.add_argument('--nproc',
                    type = int,
                    default = 8)
    
    args = CLI.parse_args()
    print(args)
    
    # Specify base simulation folder ------
    if args.machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle",
                                         "rb"))[args.method]
        method_folder = method_params['method_folder_x7']

    if args.machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", 
                                         "rb"))[args.method]
        method_folder = method_params['method_folder']
    
    # Speficy names of process parameters
    process_params = method_params['param_names'] + method_params['boundary_param_names']
    
    # Make output folder if it doesn't exist
    if not os.path.isdir(method_folder + args.outfolder + '/'):
        os.mkdir(method_folder + args.outfolder + '/')

    # Main function 
    kde_util.kde_from_simulations_fast_parallel(base_simulation_folder = method_folder + args.simfolder,
                                                file_name_prefix = args.fileprefix,
                                                file_id = args.fileid,
                                                target_folder = method_folder + args.outfolder,
                                                n_by_param = args.nbyparam,
                                                mixture_p = args.mixture,
                                                process_params = process_params,
                                                print_info = False,
                                                n_processes= args.nproc)
    
# UNUSED --------------------------
    # LBA
#     process_params = ['v_0', 'v_1', 'A', 'b', 's', 'ndt']
#     files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

#     # DDM NDT
#     process_params = ['v', 'a', 'w', 'ndt']
#     files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

    # DDM ANGLE NDT
    #process_params = ['v', 'a', 'w', 'ndt', 'theta']
    #files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))
