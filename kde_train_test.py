import ddm_data_simulation as ddm_sim
import scipy as scp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import multiprocessing as mp
import psutil
import pickle
import os

import kde_training_utilities as kde_util
import kde_class as kde


if __name__ == "__main__":

    # PICK
    base_simulation_folder = '/users/afengler/data/kde/ddm/base_simulations_ndt_20000/'
    target_folder = '/users/afengler/data/kde/ddm/train_test_data_ndt_20000'
    
    # LBA
#     process_params = ['v_0', 'v_1', 'A', 'b', 's', 'ndt']
#     files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

    # DDM NDT
    process_params = ['v', 'a', 'w', 'ndt']
    files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    kde_util.kde_from_simulations(base_simulation_folder = base_simulation_folder,
                                  target_folder = target_folder,
                                  n_total = 1000000,
                                  mixture_p = [0.8, 0.1, 0.1],
                                  process_params = process_params,
                                  print_info = False,
                                  files_ = files_,
                                  p_files = 0.01)