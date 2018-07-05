# Loading necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as scp
import matplotlib.pyplot as plt
import shutil
import csv
import os
import dnnregressor_model_and_input_fn as dnnreg_model_input

# Get working directory (in case it's useful)
cwd = os.getcwd()


# Suppressing tensorflow output to keep things clean at this point
tf.logging.set_verbosity(tf.logging.ERROR) # could be tf.logging.ERROR, tf.logging.INFO, tf.logging.DEBUG, tf.logging.FATAL, tf.logging.WARN ....

def get_dnnreg_predictor(model_directory = '', params = []):
    dnnregressor  = tf.estimator.Estimator(model_fn = dnnreg_model_input.dnn_regressor,
                                           model_dir = model_directory,
                                           params = params)

    return dnnregressor

def get_predictions(regressor = [],
                    features = [],
                    labels = []):
    predictions = pd.DataFrame(list(regressor.predict(
                                       input_fn = lambda: dnnreg_model_input.eval_input_fn(features = features,
                                                                         labels = labels,
                                                                         batch_size = 1,
                                                                         num_epochs = 1),
                                       predict_keys = 'output',
                                       yield_single_examples = False,
                                      )))
    return predictions
