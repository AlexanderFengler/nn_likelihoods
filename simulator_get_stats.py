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

# My own code
import kde_class as kde
import ddm_data_simulation as ddm_simulator 
import boundary_functions as bf
import kde_training_utilities as kde_utils

if __name__ == "__main__":
    # Specify base simulation folder
#     base_simulation_folder = '/users/afengler/data/kde/ddm/base_simulations_ndt_20000/'
    
    # PARAM RANGES: LINEAR COLLAPSE
    # ORNSTEIN UHLENBECK
#     param_ranges = {'a': [0.5, 2],
#                     'w': [0.3, 0.7],
#                     'v': [-2, 2],
#                     'g':[-2, 2]
# #                     'theta': [0, np.pi/2.2],
# #                     'node': [0, 5]
#                     }

#     # LBA
#     param_ranges = {'v_0': [1, 2],
#                     'v_1': [1, 2],
#                     'A': [0, 1],
#                     'b':[1.5, 3],
#                     's':[0.1, 0.2]
#                     }
    
    # LBA NDT
#     param_ranges = {'v_0': [1, 2],
#                     'v_1': [1, 2],
#                     'A': [0, 1],
#                     'b':[1.5, 3],
#                     's':[0.1, 0.2],
#                     'ndt':[0, 1]
#                     }
#     base_simulation_folder = '/users/afengler/data/kde/lba/base_simulations_ndt_20000/'
    
    # DDM NDT
#     param_ranges = {'v': [-2.0, 2.0],
#                     'a': [0.5, 2],
#                     'w': [0.3, 0.7],
#                     'ndt': [0, 1]
#                    }
    
    # DDM ANGLE NDT
    param_ranges = {'v': [-2.0, 2.0],
                    'a': [0.5, 2],
                    'w': [0.3, 0.7],
                    'ndt': [0, 1],
                    'theta': [0, np.pi/2 - 0.2]
                    }   
    base_simulation_folder = '/users/afengler/data/kde/angle/base_simulations_ndt_20000/'
    
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