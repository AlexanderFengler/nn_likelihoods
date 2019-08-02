# Load packages
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Activations
def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

# Function to extract network architecture 
def extract_architecture(model):
    biases = []
    activations = []
    weights = []
    for layer in model.layers:
        if layer.name == "input_1":
            continue
        weights.append(layer.get_weights()[0])
        biases.append(layer.get_weights()[1])
        activations.append(layer.get_config()["activation"])
    return weights, biases, activations

# Function to perform forward pass given architecture
def predict(x, weights, biases, activations):
    # Activation dict
    activation_fns = {"relu":relu, "linear":linear}
    
    for i in range(len(weights)):
        x = activation_fns[activations[i]](
            np.dot(x, weights[i]) + biases[i])
    return x

def log_p(params, weights, biases, activations, data, ll_min = 1e-29):
    param_grid = np.tile(params, (data.shape[0], 1))
    inp = np.concatenate([param_grid, data], axis = 1)
    out = np.maximum(predict(inp, weights, biases, activations),ll_min)
    
    return -np.sum(np.log(out))