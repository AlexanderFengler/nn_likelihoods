# Basic python utilities
import numpy as np
import scipy as scp 
from scipy.stats import gamma

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
#import argparse

# My own code
import kde_class as kde
import ddm_data_simulation as ddm_simulator 
import boundary_functions as bf
import kde_training_utilities as kde_utils

if __name__ == "__main__":
    # Specify base simulation folder
    base_simulation_folder = '/home/afengler/git_repose/nn_likelihoods/data_storage/kde/linear_collapse/base_simulations/'
    
    # PARAM RANGES: LINEAR COLLAPSE
    param_ranges = {'a': [0.5, 2],
                    'w': [0.3, 0.7],
                    'v': [-1, 1],
                    'theta': [0, np.pi/2.2],
                    'node': [0, 5]}
    
    # FILTERS: GENERAL
    filters = {'mode': 20, # != 
               'choice_cnt': 10, # > 
               'mean_rt': 15, # < 
               'std': 0, # > 
               'mode_cnt_rel': 0.5  # < 
              }
    
    # Run filter
    kde_utils.filter_simulations(base_simulation_folder = base_simulation_folder, 
                                 param_ranges = param_ranges,
                                 filters = filters)