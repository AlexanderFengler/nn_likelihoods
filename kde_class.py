# KDE GENERATORS
import numpy as np
import scipy as scp
import time
from datetime import datetime
from sklearn.neighbors import KernelDensity


# Generate class for log_kdes
class logkde():
    def __init__(self,
                 simulator_data, # Simulator_data is the kind of data returned by the simulators in ddm_data_simulatoin.py
                 bandwidth_type = 'silverman',
                 auto_bandwidth = True):

        self.attach_data_from_simulator(simulator_data)
        self.generate_base_kdes(auto_bandwidth = auto_bandwidth,
                                bandwidth_type = bandwidth_type)
        self.simulator_info = simulator_data[3]
        #self.choice_options = np.unique(simulator_data[1]) # storing choice options of process that generated data
        #self.choice_options.sort()
        #self.process_params = simulator_data[2] # storing the process parameters underlying the input data

        #self.data = (([0, 2, 4], 1, 0.5),
        #             ([1, 2, 3], -1, 0.5))
        #self.base_kdes = 'no kde assigned yet'
        #self.bandwidths = 'no bandwidth produced yet'

    # Function to compute bandwidth parameters given data-set
    # (At this point using Silverman rule)
    def compute_bandwidths(self,
                           type = 'silverman'):

        self.bandwidths = []
        if type == 'silverman':
            for i in range(0, len(self.data['choices']), 1):
                self.bandwidths.append(bandwidth_silverman(sample = np.log(self.data['rts'][i])))

    # Function to generate basic kdes
    # I call the function generate_base_kdes because in the final evaluation computations
    # we adjust the input and output of the kdes appropriately (we do not use them directly)
    def generate_base_kdes(self,
                           auto_bandwidth = True,
                           bandwidth_type  = 'silverman'):

        # Compute bandwidth parameters
        if auto_bandwidth:
            self.compute_bandwidths(type = bandwidth_type)

        # Generate the kdes
        self.base_kdes = []
        for i in range(0, len(self.data['choices']), 1):
            self.base_kdes.append(KernelDensity(kernel = 'gaussian',
                                                bandwidth = self.bandwidths[i]).fit(np.log(self.data['rts'][i])))

    # Function to evaluate the kde log likelihood at chosen points
    def kde_eval(self,
                 where = np.ones((10, 1)),
                 which = [-1, 1],  #kde
                 log_eval = True):

        log_where = np.log(where)
        log_kde_eval = []
        kde_eval = []
        which_data_idx = []

        for i in range(0, len(self.data['choices']), 1):
            if self.data['choices'][i] in which:
                which_data_idx.append(i)

        if log_eval:
            for i in which_data_idx:
                log_kde_eval.append(np.log(self.data['choice_proportions'][i]) + self.base_kdes[i].score_samples(log_where) - log_where[:, 0])
            return log_kde_eval, which_data_idx
        else:
            for i in which_data_idx:
                kde_eval.append(np.exp(np.log(self.data['choice_proportions'][i]) + self.base_kdes[i].score_samples(log_where) - log_where[:, 0]))
            return kde_eval, which_data_idx
        # return kde (log) likelihoods and also the indices in self.data corresponding to the choices requested

    def kde_sample(self,
                   n_samples = 2000,
                   which = [-1, 1]):

        which_data_idx = []
        kde_samples = []

        for i in range(0, len(self.data['choices']), 1):
            if self.data['choices'][i] in which:
                which_data_idx.append(i)

        for i in which_data_idx:
            kde_samples.append(np.exp(self.base_kdes[i].sample(n_samples = n_samples)))

        return kde_samples, which_data_idx
        # return samples from the kde model and also the indices in self.data corresponding to the choices requested

    # Helper function to transform ddm simulator output to dataset suitable for
    # the kde function class
    def attach_data_from_simulator(self,
                                   simulator_data = ([0, 2, 4], [-1, 1, -1])):

        choices = np.unique(simulator_data[1])
        n = len(simulator_data[0])
        self.data = {'rts': [],
                     'choices': [],
                     'choice_proportions': []}

        # Loop through the choices made to get proportions and separated out rts
        for c in choices:
            self.data['choices'].append(c)
            rts_temp = np.expand_dims(simulator_data[0][simulator_data[1] == c], axis = 1)
            prop = len(rts_temp) / n
            self.data['rts'].append(rts_temp)
            self.data['choice_proportions'].append(prop)

# Support functions (accessible from outside the main class defined in script)
def bandwidth_silverman(sample = [0,0,0]):
    std = np.std(sample)
    n = len(sample)
    return np.power((4/3), 1/5) * std * np.power(n, (-1/5))
