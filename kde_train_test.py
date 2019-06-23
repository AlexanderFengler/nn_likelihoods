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
import regit

import kde_training_utilities as kde_util
import kde_class as kde


if __name__ == "__main__":

    # PICK
    base_simulation_folder = '/users/afengler/data/kde/weibull/base_simulations/'
    target_folder = '/users/afengler/data/kde/weibull/train_test_data/'
    process_params = ['v', 'a', 'w', 'node', 'shape', 'scale']
    files_ = pickle.load( open('/users/afengler/data/kde/weibull/base_simulations/keep_files.pickle', 'rb'))

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    kde_util.kde_from_simulations(base_simulation_folder = base_simulation_folder,
                                  target_folder = target_folder,
                                  n_total = 10000000,
                                  mixture_p = [0.8, 0.1, 0.1],
                                  process_params = process_params,
                                  print_info = False,
                                  target_file_format = 'pickle',
                                  files_ = files_,
                                  p_files = 0.1)
