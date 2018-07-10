import numpy as np
import scipy as scp
import os
import ddm_data_simulation
import make_data_wfpt

class ddm_mle_estimator:
    def __init__(self):
        self.ddm_params = dict({'v': 1,
                            'a': 1,
                            'w': 0.5,
                            's': 1,
                            'delta_t': 0.001,
                            'max_t': 20})
        self.data = []
    def make_data(self, n_samples = 20000):
        self.data = ddm_data_simulation.ddm_simulate_rts(v = self.ddm_params['v'],
                                                        a = self.ddm_params['a'],
                                                        w = self.ddm_params['w'],
                                                        s = self.ddm_params['s'],
                                                        delta_t = self.ddm_params['delta_t'],
                                                        max_t = self.ddm_params['max_t'])



# v = 0, # drift by timestep 'delta_t'
#                       a = 1, # boundary separation
#                       w = 0.5,  # between -1 and 1
#                       s = 1, # noise sigma
#                       delta_t = 0.001,
#                       max_t = 20,
#                       n_samples = 20000
