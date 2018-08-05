# Loading necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as scp
import time
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
import shutil
import csv
import os
from shutil import copyfile
import glob

# Importing the helper functions to generate my dnnregressor
import dnnregressor_model_and_input_fn as dnnreg_model_input

# Importing helper functions to make data set (CHANGE WHEN CHANGING DATA GENERATING MODEL)
# import make_data_sin as mds
import make_data_wfpt as mdw

# Getting working directory in case it is useful later (definition of model_dir)
cwd = os.getcwd()

# Suppressing tensorflow output to keep things clean at this point
tf.logging.set_verbosity(tf.logging.INFO) # could be tf.logging.ERROR, tf.logging.INFO, tf.logging.DEBUG, tf.logging.FATAL, tf.logging.WARN ....

# Functions that generates a panda dataframe with rows as sets of
# hyperparameters --> passed to run_training(...)
def make_hyper_params_csv(hyp_hidden_units = [[i, j] for i in [10, 300, 500] for j in [300]],
                           hyp_activations = [[i, j] for i in ['relu'] for j in ['relu']],
                           hyp_optimizer = ['momentum', 'adagrad'],  #['momentum', 'sgd'],
                           hyp_learning_rate = [0.01, 0.005, 0.001],  #  [0.001, 0.005, 0.01, 0.02]
                           hyp_loss_fn = ['mse', 'mae'],  #['mse', 'abs'],
                           hyp_l_1 =  [0.0, 0.5],  # [0.0, 0.1, 1.0, 10.0]
                           hyp_l_2 =  [0.0, 0.5],  # [0.0, 0.1, 0.5, 1.0]
                           hyp_batch_size = [1000, 10000],  #[1, 10, 100, 500, 1000, 10000]
                           out_file = 'hyper_parameters.csv'):
    hyper_params = pd.DataFrame(columns=['hidden_units',
                                     'activations',
                                     'optimizer',
                                     'learning_rate',
                                     'loss_fn',
                                     'l_1',
                                     'l_2',
                                     'batch_size'])

    cnt = 0
    for tmp_hidden_units in hyp_hidden_units:
        for tmp_activations in hyp_activations:
            for tmp_optimizer in hyp_optimizer:
                for tmp_learning_rate in hyp_learning_rate:
                    for tmp_loss_fn in hyp_loss_fn:
                        for tmp_l_1 in hyp_l_1:
                            for tmp_l_2 in hyp_l_2:
                                for tmp_batch_size in hyp_batch_size:
                                    hyper_params.loc[cnt] = [tmp_hidden_units,
                                                            tmp_activations,
                                                            tmp_optimizer,
                                                            tmp_learning_rate,
                                                            tmp_loss_fn,
                                                            tmp_l_1,
                                                            tmp_l_2,
                                                            tmp_batch_size]
                                    cnt += 1
    # Write to file
    hyper_params.to_csv(out_file)
    #return hyper_params

def run_training(hyper_params = [],
                 headers = [],
                 training_features = [],
                 training_labels = [],
                 feature_columns = [],
                 model_directory = '...',
                 max_epoch = 5000,
                 eval_after_n_epochs = 10,
                 print_info = True):

    # Training loop (instead of nested for loops, for loops through panda rows?!)
    # Initializing metrics as some big numbers
    old_metric = 1e10
    new_metric = 1e10 - 1

    # Get time
    start_time = time.time()

    # Store hyper_parameters into our model parameter dictionary
    model_params = {
                    'feature_columns': feature_columns,
                    'hidden_units': hyper_params['hidden_units'][0],
                    'activations': hyper_params['activations'][0],
                    'optimizer': hyper_params['optimizer'][0],
                    'learning_rate': hyper_params['learning_rate'][0],
                    'loss_fn': hyper_params['loss_fn'][0],
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'rho': 0.9,
                    'l_1': hyper_params['l_1'][0], # NOTE: Cannot supply integers here
                    'l_2': hyper_params['l_2'][0], # NOTE: Cannot supply integers here
                    'batch_size': hyper_params['batch_size'][0]
                    }

    # Define number of training steps before evaluation
    # I want to evaluate after every 10 epochs
    steps_until_eval = eval_after_n_epochs * (len(training_labels) // hyper_params['batch_size'][0])

    date_time_str = datetime.now().strftime('_%m_%d_%y_%H_%M_%S')
    basedir = model_directory + '/dnnregressor_' + model_params['loss_fn'] + date_time_str

    # Making the estimator
    dnnregressor = tf.estimator.Estimator(
                                         model_fn = dnnreg_model_input.dnn_regressor,
                                         params = model_params,
                                         model_dir = basedir
                                         )

    epoch_cnt = 0
    training_steps_cnt = 0

    print('max_epoch: ', max_epoch)
    print('epoch_cnt: ', epoch_cnt)
    print(new_metric < old_metric)

    while new_metric < old_metric and epoch_cnt < max_epoch:
        print('now starting training')
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

        new_metric = evaluation_metrics_test[model_params['loss_fn']]
        training_steps_cnt += steps_until_eval

        # Generate summary of hyperparameters and append results to file
        current_eval_data = [
                             str(model_params['hidden_units']),
                             str(model_params['activations']),
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

        if epoch_cnt == 0:
            with open(basedir + '/dnn_training_results'  + date_time_str + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        with open(basedir + '/dnn_training_results' + date_time_str + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(current_eval_data)

        epoch_cnt += eval_after_n_epochs
        print(str(epoch_cnt) + ' epochs run')

        if epoch_cnt >= max_epoch:
            print('training of model terminated due to reaching max epoch: ', max_epoch)


    print('skipped training')

# MAKE EXECUTABLE FROM COMMAND LINE
if __name__ == "__main__":
    # Headers for training results table
    headers = [
                 'hidden_units',
                 'activations',
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

    # Generating training and test test
    # train_features, train_labels, test_features, test_labels = mds.train_test_split(features = features,
    #                                                                             labels = labels,
    #                                                                             p = 0.8)

    print('reading in training and test set....')
    train_features, train_labels, test_features, test_labels = mdw.train_test_from_file(fname_test = '',
                                                                                        fname_train = '',
                                                                                        n = 5000000)

    # Potentially remove 'choice' column and make reaction times positive and negative at this point
    # Hyperparameters under consideration
    print('defining hyperparameters...')
    hyp_hidden_units = [[i, j] for i in [300] for j in [300]]
    hyp_activations = [[i, j] for i in ['relu'] for j in ['relu']]
    hyp_optimizer = ['adam'] #, 'adagrad'] #['momentum', 'adam']  #['momentum', 'sgd'],
    hyp_learning_rate = [0.005] #, 0.005] #[0.01, 0.005, 0.001]  #  [0.001, 0.005, 0.01, 0.02]
    hyp_loss_fn = ['mse'] #, 'mae']  #['mse', 'abs'],
    hyp_l_1 =  [0.0] # 0.5]  # [0.0, 0.1, 1.0, 10.0]
    hyp_l_2 =  [0.0] # 0.5]  # [0.0, 0.1, 0.5, 1.0]
    hyp_batch_size = [10000] # , 10000]  #[1, 10, 100, 500, 1000, 10000]

    # Make table to hyperparameters that we consider in training (WRITE TO FILE)
    make_hyper_params_csv(hyp_hidden_units = hyp_hidden_units,
                          hyp_activations = hyp_activations,
                          hyp_optimizer = hyp_optimizer,
                          hyp_learning_rate = hyp_learning_rate,    # [0.001, 0.005, 0.01, 0.02]
                          hyp_loss_fn = hyp_loss_fn,
                          hyp_l_1 =  hyp_l_1,  # [0.0, 0.1, 1.0, 10.0]
                          hyp_l_2 =  hyp_l_2, # [0.0, 0.1, 0.5, 1.0]
                          hyp_batch_size = hyp_batch_size # [1, 10, 100, 500, 1000, 10000]
                          )

    hyper_params = pd.read_csv('hyper_parameters.csv',
                               converters = {'hidden_units':eval,
                                             'activations':eval})

    # Generate feature
    feature_columns = dnnreg_model_input.make_feature_columns_numeric(features = train_features)

    # Run training across hyperparameter setups
    print('starting training....')

    run_training(hyper_params = hyper_params,
                 headers = headers,
                 training_features = train_features,
                 training_labels = train_labels,
                 feature_columns = feature_columns,
                 model_directory = model_directory,
                 max_epoch = 10000,
                 eval_after_n_epochs = 50,
                 print_info = True)
