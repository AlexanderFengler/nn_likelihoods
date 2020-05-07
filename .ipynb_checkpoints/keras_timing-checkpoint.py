import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
import keras_to_numpy as ktnp
import pickle
import cddm_data_simulation as cds
import boundary_functions as bf
import numpy as np

my_dir = '/users/afengler/OneDrive/project_nn_likelihoods/data/kde/ddm/keras_models/dnnregressor_ddm_03_29_20_17_38_58/'

biases = pickle.load(open(my_dir + 'biases.pickle', 'rb'))
weights = pickle.load(open(my_dir + 'weights.pickle', 'rb'))
activations = pickle.load(open(my_dir + 'activations.pickle', 'rb'))

# Make toy dataset
out = cds.ddm_flexbound(n_samples = 4096,
                        boundary_fun = bf.constant,
                        boundary_multiplicative = True)
out = np.concatenate([out[0], out[1]], axis = 1)

params_rep = [0, 1, 0.5, 0.]
keras_input_batch = np.zeros((out.shape[0], 6))
keras_input_batch[:, :4] = params_rep
keras_input_batch[:, 4:] = out

def mlp_target(params, data, ll_min = -16.11809):
    mlp_input_batch = np.zeros((data.shape[0], 6))
    mlp_input_batch[:, :4] = params
    mlp_input_batch[:, 4:] = data
    out = np.maximum(ktnp.predict(mlp_input_batch, weights, biases, activations, n_layers = 4), ll_min)
    return np.sum(out)

class mlp_target_class:
    def __init__(self, 
                 data = [],
                 weights = [],
                 biases = [],
                 activations = [],
                 ll_min = -16.11809,
                 n_params = 4):
        
        self.n_params = n_params
        self.data = data
        self.ll_min = ll_min
        self.batch = np.zeros((self.data.shape[0], n_params + 2))
        self.batch[:, self.n_params:] = data
        self.weights = weights
        self.biases = biases
        self.activations = activations
        
    
    def target(self, 
               params):
        self.batch[:, :self.n_params] = np.tile(params, (self.data.shape[0], 1))
        return np.sum(np.maximum(ktnp.predict(self.batch, self.weights, self.biases, self.activations, n_layers = 4), self.ll_min))
    
%%timeit -n 1 -r 100 
mlp_target(params_rep, out)

%%timeit -n 1 -r 100
mlp.target(params_rep)

model = keras.models.load_model(my_dir + 'model_final.h5' )

%%timeit -n 1 -r 100
model.predict(keras_input_batch)