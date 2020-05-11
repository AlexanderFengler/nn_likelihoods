import tensorflow as tf
from tensorflow import keras
import os
import re
from datetime import datetime
import argparse
import keras_to_numpy as ktnp
import pickle
import cddm_data_simulation as cds
from cdwiener import batch_fptd
import boundary_functions as bf
import numpy as np
import yaml

# Define mlp class for numpy forward pass
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
        

def nf_target(params, data, likelihood_min = 1e-10):
    return np.sum(np.maximum(np.log(batch_fptd(data[:, 0] * data[:, 1] * (- 1),
                                               params[0],
                                               params[1] * 2, 
                                               params[2],
                                               params[3])),
                                               np.log(likelihood_min)))
    

# INITIALIZATIONS -------------------------------------------------------------
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--method",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--nsamples",
                     nargs = "*",
                     type = int,
                     default = 1000)
    CLI.add_argument("--nreps",
                     type = int,
                     default = 100)
    
    args = CLI.parse_args()
    print(args)

    machine = args.machine
    method = args.method
    nsamples = args.nsamples
    nreps = args.nreps

    # Get (machine dependent) network directory 
    if machine == 'x7':
        with open("model_paths_x7.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)[method]
            network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

            print('Loading network from: ')
            print(network_path)
            # model = load_model(network_path + 'model_final.h5', custom_objects = {"huber_loss": tf.losses.huber_loss})

    if machine == 'ccv':
        with open("model_paths.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)[method]
            network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
                
            print('Loading network from: ')
            print(network_path)

    # Load network parameters in
    biases = pickle.load(open(network_path + 'biases.pickle', 'rb'))
    weights = pickle.load(open(network_path + 'weights.pickle', 'rb'))
    activations = pickle.load(open(network_path + 'activations.pickle', 'rb'))
    
    # Load keras model
    keras_model = keras.models.load_model(network_path + 'model_final.h5')

    info = {}
    info['numpy_timings'] = []
    info['keras_var_batch_timings'] = []
    info['keras_fix_batch_timings'] = []
    info['keras_no_batch_timings'] = []
    info['navarro_timings'] = []
    info['nsamples'] = []

    # Run timingss
    for n in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        
        print('nsamples: ', n)
        # Generate toy dataset
        out = cds.ddm_flexbound(n_samples = n,
                                boundary_fun = bf.constant,
                                boundary_multiplicative = True)
        out = np.concatenate([out[0], out[1]], axis = 1)
        
        # Arbitraty parameters vector
        params_rep = [0, 1, 0.5, 0.]

        # Prepare batch matrix for keras model
        keras_input_batch = np.zeros((out.shape[0], 6))
        keras_input_batch[:, 4:] = out

        
        # Navarro Fuss timings
        print('Running Navarro')
        for i in range(nreps):
            start = datetime.now()
            nf_target(params_rep, data, likelihood_min = 1e-10)
            info['navarro_timings'].append((datetime.now() - start).total_seconds())
            info['nsamples'].append(n)
            
        # Numpy timings
        print('Running numpy')
        for i in range(nreps):
            # Load numpy model
            numpy_model = mlp_target_class(data = out,
                                           weights = weights,
                                           biases = biases,
                                           activations = activations)
            start = datetime.now()
            numpy_model.target(params_rep)
            info['numpy_timings'].append((datetime.now() - start).total_seconds())

        
        # Keras timings variable batch size
        print('Running keras var batch')
        for i in range(nreps):
            start = datetime.now()
            keras_input_batch[:, :4] = params_rep
            keras_model.predict(keras_input_batch, batch_size = nsamples)
            info['keras_var_batch_timings'].append((datetime.now() - start).total_seconds())

        # Keras timings fixed batch size
        print('Running keras fix batch')
        for i in range(nreps):
            start = datetime.now()
            keras_input_batch[:, :4] = params_rep
            keras_model.predict(keras_input_batch, batch_size = 1024)
            info['keras_fix_batch_timings'].append((datetime.now() - start).total_seconds())

        # Keras timings unspecified batch size
        print('Running keras no batch')
        for i in range(nreps):
            start = datetime.now()
            keras_input_batch[:, :4] = params_rep
            keras_model.predict(keras_input_batch)
            info['keras_no_batch_timings'].append((datetime.now() - start).total_seconds())

    pickle.dump(info, open('/users/afengler/data/timings/timings.pickle', 'wb'), protocol = 4)