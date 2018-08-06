# Loading necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as scp
import time
from datetime import datetime
import csv
import os

# Importing the helper functions to generate my dnnregressor
import dnnregressor_model_and_input_fn as dnnreg_model_input

# Importing helper functions to make data set (CHANGE WHEN CHANGING DATA GENERATING MODEL)
# import make_data_sin as mds
import make_data_wfpt as mdw
import make_data_sin as mds

# Getting working directory in case it is useful later (definition of model_dir)
cwd = os.getcwd()

# Suppressing tensorflow output to keep things clean at this point
tf.logging.set_verbosity(tf.logging.INFO) # could be tf.logging.ERROR, tf.logging.INFO, tf.logging.DEBUG, tf.logging.FATAL, tf.logging.WARN ....

# Functions that generates a panda dataframe with rows as sets of
# hyperparameters --> passed to run_training(...)

def run_training(hyper_params = [],
                 headers = [],
                 training_features = [],
                 training_labels = [],
                 feature_columns = [],
                 model_directory = '...',
                 eval_after_n_epochs = 10,
                 print_info = True):

    # Training loop (instead of nested for loops, for loops through panda rows?!)
    # Initializing metrics as some big numbers
    old_metric = 1e10
    new_metric = 1e10 - 1

    # Get time
    start_time = time.time()

    # Store hyper_parameters into our model parameter dictionary
    model_params = hyper_params

    # Define number of training steps before evaluation
    # I want to evaluate after every 10 epochs
    steps_until_eval = model_params['eval_after_n_epochs'] * (len(training_labels) //model_params['batch_size'])

    # Get current date and time for file naming purposes
    date_time_str = datetime.now().strftime('_%m_%d_%y_%H_%M_%S')

    # Specify folder for storage of graph and results
    basedir = model_directory + '/dnnregressor_' + model_params['loss_fn'] + date_time_str

    # Making the estimator
    dnnregressor = tf.estimator.Estimator(
                                         model_fn = dnnreg_model_input.dnn_regressor,
                                         params = model_params,
                                         model_dir = basedir
                                         )


    # Initialze epoch and training steps counters
    max_epoch = model_params['max_epoch']
    epoch_cnt = 0
    training_steps_cnt = 0


    # Training loop
    print('now starting training')
    while new_metric < old_metric and epoch_cnt < max_epoch:

        # Update reference metric
        old_metric = new_metric

        # Training the estimator
        dnnregressor.train(
                           input_fn = lambda: dnnreg_model_input.train_input_fn(features = train_features,
                                                       labels = train_labels,
                                                       batch_size = model_params['batch_size']),
                           steps = steps_until_eval
                          )

        # Evaluation Metrics
        evaluation_metrics_test = dnnregressor.evaluate(
                                                        input_fn = lambda: dnnreg_model_input.eval_input_fn(features = test_features,
                                                                                                            labels = test_labels,
                                                                                                            batch_size = 100,
                                                                                                            num_epochs = 1
                                                                                                            )
                                                        )

        # Update metric
        new_metric = evaluation_metrics_test[model_params['loss_fn']]
        training_steps_cnt += steps_until_eval
        print('Old metric value:' + str(old_metric))
        print('New metric value:' + str(new_metric))
        print('Improvement in evaluation metric: ' + str(old_metric - new_metric))

        # Generate summary of hyperparameters and append results to file
        current_eval_data = [
                             str(model_params['hidden_units']),
                             str(model_params['activations']),
                             model_params['output_activation'],
                             model_params['optimizer'],
                             str(model_params['learning_rate']),
                             model_params['loss_fn'],
                             str(model_params['beta1']),
                             str(model_params['beta2']),
                             str(model_params['rho']),
                             str(model_params['l_1']),
                             str(model_params['l_2']),
                             str(model_params['batch_size']),
                             str(evaluation_metrics_test['mse']),
                             str(evaluation_metrics_test['mae']),
                             str(steps_until_eval),
                             '%.1f' % (time.time() - start_time),
                             date_time_str
        ]

        # If first round of training, initialize training_results table with headers
        if epoch_cnt == 0:
            with open(basedir + '/dnn_training_results'  + date_time_str + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        # Update training_results table with current training results
        with open(basedir + '/dnn_training_results' + date_time_str + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(current_eval_data)

        # Update epoch_cnt and print current number of epochs completed
        epoch_cnt += model_params['eval_after_n_epochs']
        print(str(epoch_cnt) + ' of a maximum of ' + str(max_epoch) +  ' epochs run')

        # Print info if training terminated due to reaching max_epoch as passed to the function
        if epoch_cnt >= max_epoch:
            print('training of model terminated due to reaching max epoch: ', max_epoch)

    # Report that training is finished once we exited the while loop
    print('finished training')

# MAKE EXECUTABLE FROM COMMAND LINE
if __name__ == "__main__":
    # Headers for training results table
    headers = [
                 'hidden_units',
                 'activations',
                 'output_activation',
                 'optimizer',
                 'learning_rate',
                 'loss_fn',
                 'beta1',
                 'beta2',
                 'rho',
                 'l_1',
                 'l_2',
                 'batch_size',
                 'mse_test',
                 'mae_test',
                 'training_steps',
                 'training_time',
                 'time_started'
    ]

    # Define directory for models to be spit out into
    model_directory = cwd + '/tensorflow_models'

    # Potentially remove 'choice' column and make reaction times positive and negative at this point
    # Hyperparameters under consideration
    print('defining hyperparameters...')

    hyper_params = { 'hidden_units': [300,300],
                     'activations': ['sigmoid', 'sigmoid'],
                     'output_activation': 'linear',
                     'optimizer': 'adam',
                     'learning_rate': 0.005,
                     'loss_fn': 'mse',
                     'beta1': 0.9,
                     'beta2': 0.999,
                     'rho': 0.9,
                     'l_1': 0.0, # NOTE: Cannot supply integers here
                     'l_2': 0.0, # NOTE: Cannot supply integers here
                     'batch_size': 10000,
                     'max_epoch': 10000,
                     'eval_after_n_epochs': 100,
                     'training_data_size': 5000000,
                     'data_type': 'wfpt'
                    }

    # mini sanity check to make sure that batch_size is smaller or equal than training_data_size
    assert hyper_params['batch_size'] <= hyper_params['training_data_size'], 'Make batch size smaller or equal to training data size please..... (specified in hyper_params dictionary)'

    # Reading in training data
    if hyper_params['data_type'] == 'wfpt':
        print('reading in training and test set....')
        train_features, train_labels, test_features, test_labels = mdw.train_test_from_file(fname_test = '',
                                                                                            fname_train = '',
                                                                                            n = hyper_params['training_data_size'])

    if hyper_params['data_type'] == 'sin':
        features, labels = mds.make_data()
        train_features, train_labels, test_features, test_labels = mds.train_test_split(features, labels, p = 0.8)

    # Generate feature
    feature_columns = dnnreg_model_input.make_feature_columns_numeric(features = train_features)

    hyper_params['feature_columns'] = feature_columns

    # Run training across hyperparameter setups
    print('starting training....')

    run_training(hyper_params = hyper_params,
                 headers = headers,
                 training_features = train_features,
                 training_labels = train_labels,
                 feature_columns = feature_columns,
                 model_directory = model_directory,
                 print_info = True)
