import ddm_data_simulation as ddm_sim
import scipy as scp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
#import kde_class
import multiprocessing as mp
import psutil
import pickle 
import os
import re

import kde_training_utilities as kde_util
import kde_class as kde


if __name__ == "__main__":
    
    idx = 1
    base_simulation_folder = '/users/afengler/data/kde/weibull/base_simulations/'
    target_folder = '/users/afengler/data/kde/weibull/train_test_data_' + str(idx) + '/'
    process_params = ['v', 'a', 'w', 'node', 'shape', 'scale'] 
    
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
        
    kde_util.kde_train_test_from_simulations_flexbound(base_simulation_folder = base_simulation_folder,
                                                       target_folder = target_folder,
                                                       n_total = 55000000,
                                                       p_train = 0.9,
                                                       mixture_p = [0.8, 0.1, 0.1],
                                                       process_params = process_params,
                                                       model = 'ddm_weibull',
                                                       print_info = False,
                                                       target_file_format = 'pickle',
                                                       n_files_max = 'all')
    