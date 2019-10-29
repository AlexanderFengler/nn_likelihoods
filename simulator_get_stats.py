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
import sys

# My own code
import kde_class as kde
import ddm_data_simulation as ddm_simulator 
import boundary_functions as bf
import kde_training_utilities as kde_utils

if __name__ == "__main__":
    # Specify base simulation folder
   
    # DDM ANGLE NDT
    param_ranges = {'v': [-2.0, 2.0],
                    'a': [0.5, 2],
                    'w': [0.3, 0.7],
                    'ndt': [0, 1],
                    'theta': [0, np.pi/2 - 0.2]
                    }  
    
    base_simulation_folder = '/users/afengler/data/kde/weibull_cdf/base_simulations_ndt_20000/'
    file_name_prefix = 'weibull_cdf_ndt_base_simulations'
    file_id = sys.argv[1]
    
    # FILTERS: GENERAL
    filters = {'mode': 20, # != 
               'choice_cnt': 10, # > 
               'mean_rt': 15, # < 
               'std': 0, # > 
               'mode_cnt_rel': 0.5  # < 
              }
    
    # Run filter new
    kde_utils.filter_simulations_fast(base_simulation_folder = base_simulation_folder,
                                      file_name_prefix = file_name_prefix,
                                      file_id = file_id,
                                      param_ranges = 'none',
                                      filters = filters)
    # Run filter
#     kde_utils.filter_simulations(base_simulation_folder = base_simulation_folder, 
#                                  param_ranges = param_ranges,
#                                  filters = filters)