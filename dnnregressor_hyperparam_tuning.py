# Loading necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as scp
import time
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
import csv
import os

# Importing the helper functions to generate my dnnregressor
import dnnregressor_model_and_input_fn as dnnreg_model_input

# Importing helper functions to make data set (CHANGE WHEN CHANGING DATA GENERATING MODEL)
# import make_data_sin as mds
import make_data_wfpt as mdw

# Getting working directory in case it is useful later (definition of model_dir)
cwd = os.getcwd()

# Suppressing tensorflow output to keep things clean at this point
tf.logging.set_verbosity(tf.logging.INFO) # could be tf.logging.ERROR, tf.logging.INFO, tf.logging.DEBUG, tf.logging.FATAL, tf.logging.WARN ....

# Function that sets up a csv file with headers in which
# we collect training results for models under consideration
def generate_summary_csv_headers(headers = [], file = 'dnnregressor_result_table.csv'):
    with open('dnnregressor_result_table.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

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

# Function that gets the best n hyperparameter setups collected from a round of training in
# the summary csv file generated (supplied as result_file)
def get_best_hyperparams(n_models_to_consider = 10, # specify number of columns (hyperparameter setups) to return
                         input_file = 'dnnregressor_best_hyperparams_table.csv', # specify file that contains training results
                         metric = 'mse' # specify the column by which to sort (should be a metric that was considered in training)
                        ):
    model_results = pd.read_csv(input_file,
                                converters = {'hidden_units':eval,
                                              'activations':eval})
    model_results = model_results.loc[model_results['loss_fn'] == metric].copy()
    model_results = model_results.reset_index()
    model_results = model_results.sort_values(by = metric + '_test')
    model_results = model_results.reset_index()

    if model_results.shape[0] < 10:
        n_models_to_consider = model_results.shape[0]
    return model_results.iloc[0:n_models_to_consider].copy()

# Function that trains dnn_regressor over all hyper parameter setups (hyper_params.. panda dataframe) supplied

# NOTE: Training steps we adjust adaptively s.t. we don't increase further if the error of the last
def run_training(hyper_params = [],
                 save_models = False,
                 headers = [],
                 training_features = [],
                 training_labels = [],
                 feature_columns = [],
                 model_directory = '...',
                 max_epoch = 100000,
                 print_info = True,
                 min_training_steps = 1):
    # Training loop (instead of nested for loops, for loops through panda rows?!)
    max_idx = hyper_params.shape[0]
    cnt = 0
    for i in range(0, max_idx, 1):
        # Initializing metrics as some big numbers
        old_metric = 10000000000
        new_metric = 10000000000 - 1
        n_training_steps = min_training_steps

        while new_metric < old_metric and n_training_steps < int(((training_labels.shape[0] / hyper_params['batch_size'][i]) * max_epoch)):
            # Get time
            start_time = time.time()

            # Update training_steps
            n_training_steps  = n_training_steps * 2

            # Update metric
            old_metric = new_metric

            # Printing some info
            print( str(datetime.now()) + ', Start training of model ' + str(i) + ' of ' + str(max_idx) + ' with n_training_steps: ' + str(n_training_steps))

            # Store hyper_parameters into our model parameter dictionary
            model_params = {
                'feature_columns': feature_columns,
                'hidden_units': hyper_params['hidden_units'][i],
                'activations': hyper_params['activations'][i],
                'optimizer': hyper_params['optimizer'][i],
                'learning_rate': hyper_params['learning_rate'][i],
                'loss_fn': hyper_params['loss_fn'][i],
                'beta1': 0.9,
                'beta2': 0.999,
                'rho': 0.9,
                'l_1': hyper_params['l_1'][i], # NOTE: Cannot supply integers here
                'l_2': hyper_params['l_2'][i], # NOTE: Cannot supply integers here
                'batch_size': hyper_params['batch_size'][i],
                'n_training_steps': n_training_steps
            }

            if save_models == True:
                basedir = model_directory + '/dnnregressor_' + model_params['loss_fn'] + '_' + str(cnt)
                # when we want to fave models we want to fix the number of training steps directly instead of adapting it accordint to success (as desired in hyperparameter optimization)
                model_params['n_training_steps'] = hyper_params['n_training_steps'][i]
                old_metric = -np.inf
            else:
                basedir = model_directory + '/tmp'

            # Making the estimator
            dnnregressor = tf.estimator.Estimator(model_fn = dnnreg_model_input.dnn_regressor,
                                                 params = model_params,
                                                 model_dir = basedir
                                                 )

            # Training the estimator
            dnnregressor.train(input_fn = lambda: dnnreg_model_input.train_input_fn(features = train_features,
                                                           labels = train_labels,
                                                           batch_size = model_params['batch_size']),
                               steps = model_params['n_training_steps']
                              )

            # Evaluation Metrics
            evaluation_metrics_train = dnnregressor.evaluate(
                                                       input_fn = lambda: dnnreg_model_input.eval_input_fn(features = train_features,
                                                                                         labels = train_labels,
                                                                                         batch_size = len(train_features) + 1,
                                                                                         num_epochs = 1)
                                                      )

            evaluation_metrics_test = dnnregressor.evaluate(
                                                       input_fn = lambda: dnnreg_model_input.eval_input_fn(features = test_features,
                                                                                         labels = test_labels,
                                                                                         batch_size = len(test_features) + 1,
                                                                                         num_epochs = 1)
                                                      )

            # Removing the folder that has been generated for the model
            if save_models == False:
                shutil.rmtree(basedir)

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
                                 str(evaluation_metrics_train['mse']),
                                 str(evaluation_metrics_train['mae']),
                                 str(evaluation_metrics_test['mse']),
                                 str(evaluation_metrics_test['mae']),
                                 str(n_training_steps),
                                 'model_' + str(cnt),
                                 '%.1f' % (time.time() - start_time),
                                 str(datetime.now())
            ]

            if save_models == True:
                with open(basedir + '/model_params.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

            with open('dnnregressor_result_table.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(current_eval_data)

            if save_models == True:
                with open(basedir + '/model_params.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(current_eval_data)

            new_metric = evaluation_metrics_test[model_params['loss_fn']]

            ####### JUST FOR TESTING QUICKLY IF ALL HYPERPARAMETERS WORK
            #old_metric = -np.inf
            #######
            cnt += 1
            print('Trained model with hyperparameters: ' + str(current_eval_data))

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
                 'mse_train',
                 'mae_train',
                 'mse_test',
                 'mae_test',
                 'n_training_steps',
                 'model_name',
                 'training_time',
                 'time_started'
    ]
    # Define directory for models to be spit out into
    model_directory = cwd + '/tensorflow_models'

    # Make Dataset
    # features, labels = mds.make_data()
    data = mdw.make_data(n_samples = 5000000)

    # Generating training and test test
    # train_features, train_labels, test_features, test_labels = mds.train_test_split(features = features,
    #                                                                             labels = labels,
    #                                                                             p = 0.8)

    train_features, train_labels, test_features, test_labels = mdw.train_test_split(data = data,
                                                                                    p_train = 0.8,
                                                                                    write_to_file = True)



    # Hyperparameters under consideration
    hyp_hidden_units = [[i, j] for i in [500] for j in [500]]
    hyp_activations = [[i, j] for i in ['relu'] for j in ['relu']]
    hyp_optimizer = ['sgd', 'adagrad'] #['momentum', 'adam']  #['momentum', 'sgd'],
    hyp_learning_rate = [0.01, 0.005] #[0.01, 0.005, 0.001]  #  [0.001, 0.005, 0.01, 0.02]
    hyp_loss_fn = ['mse'] #, 'mae']  #['mse', 'abs'],
    hyp_l_1 =  [0.0] # 0.5]  # [0.0, 0.1, 1.0, 10.0]
    hyp_l_2 =  [0.0, 1] # 0.5]  # [0.0, 0.1, 0.5, 1.0]
    hyp_batch_size = [1000, 10000]  #[1, 10, 100, 500, 1000, 10000]

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

    # Open training summary csv file
    generate_summary_csv_headers(headers = headers, file = 'dnnregressor_result_table.csv')

    # Generate featrue
    feature_columns = dnnreg_model_input.make_feature_columns_numeric(features = train_features)

    # Run training across hyperparameter setups
    run_training(hyper_params = hyper_params,
                 save_models = False,
                 headers = headers,
                 training_features = train_features,
                 training_labels = train_labels,
                 feature_columns = feature_columns,
                 model_directory = model_directory,
                 max_epoch = 200,
                 print_info = True,
                 min_training_steps = 200)

    # Get the best hyperparameters for further consideration
    for metric in hyp_loss_fn:
        if metric == 'mse':
            best_hyperparams_mse = get_best_hyperparams(n_models_to_consider = 10,
                                                        metric = metric,
                                                        input_file = 'dnnregressor_result_table.csv')
            best_hyperparams_mse.to_csv('dnnregressor_best_hyperparams_mse.csv')
        if metric == 'mae':
            best_hyperparams_mae = get_best_hyperparams(n_models_to_consider = 10,
                                                        metric = metric,
                                                        input_file = 'dnnregressor_result_table.csv')
            best_hyperparams_mae.to_csv('dnnregressor_best_hyperparams_mae.csv')

    # Train best models and save checkpoints

    # Best models mse
    for metric in hyp_loss_fn:
        if metric == 'mse':
            run_training(hyper_params = best_hyperparams_mse,
                         save_models = True,
                         headers = headers,
                         training_features = train_features,
                         training_labels = train_labels,
                         feature_columns = feature_columns,
                         model_directory = model_directory,
                         print_info = True)

        if metric == 'mae':
            # Best models mae
            run_training(hyper_params = best_hyperparams_mae,
                         save_models = True,
                         headers = headers,
                         training_features = train_features,
                         training_labels = train_labels,
                         feature_columns = feature_columns,
                         model_directory = model_directory,
                         print_info = True)
