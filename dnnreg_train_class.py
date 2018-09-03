# PREPARATION --------------------------

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
# -----------------------------------------

# Create class
class dnn_trainer():

    def __init__(self):
        self.headers = [
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

        self.model_directory = os.getcwd() + '/tensorflow_models'

        self.hyper_params = {
                             'hidden_units': [500,500],
                             'activations': ['relu', 'relu'],
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
                             'data_type': 'wfpt',
                             'train_test_file_signature': ''
                             }


        assert self.hyper_params['batch_size'] <= self.hyper_params['training_data_size'], 'Make batch size smaller or equal to training data size please..... (specified in hyper_params dictionary)'

    def read_in_training_data(self):
        if self.hyper_params['data_type'] == 'wfpt':
            self.train_features, self.train_labels, self.test_features, self.test_labels = mdw.train_test_from_file_rt_choice(f_signature = self.hyper_params['train_test_file_signature'],
                                                                                                                              n_samples = self.hyper_params['training_data_size']
                                                                                                                              )
        if self.hyper_params['data_type'] == 'choice_probabilities':
            self.train_features, self.train_labels, self.test_features, self.test_labels = mdw.train_test_from_file_choice_probabilities(f_signature = self.hyper_params['train_test_file_signature'],
                                                                                                                                         n_samples = self.hyper_params['training_data_size']
                                                                                                                                         )
        if self.hyper_params['data_type'] == 'sin':
            features, labels = mds.make_data()
            self.train_features, self.train_labels, self.test_features, self.test_labels = mds.train_test_split(features,
                                                                                                                labels,
                                                                                                                p = 0.8
                                                                                                                )
        # Already prepare feature columns
        feature_columns = dnnreg_model_input.make_feature_columns_numeric(features = self.train_features)
        self.hyper_params['feature_columns'] = feature_columns

    def run_training(self, print_info = True, warm_start_ckpt_path = None):

        # Get time
        start_time = time.time()

        # Store hyper_parameters into our model parameter dictionary
        model_params = self.hyper_params

        # Define number of training steps before evaluation
        # I want to evaluate after every 10 epochs
        steps_until_eval = model_params['eval_after_n_epochs'] * ((self.train_labels.shape[0]) // model_params['batch_size'])
        print('steps_until_eval:', steps_until_eval)
        # Get current date and time for file naming purposes
        date_time_str = datetime.now().strftime('%m_%d_%y_%H_%M_%S')

        # Specify folder for storage of graph and results
        basedir = self.model_directory + '/dnnregressor_' + model_params['loss_fn'] + self.hyper_params['train_test_file_signature'] + date_time_str

        # Making the estimator
        config_obj = tf.estimator.RunConfig(keep_checkpoint_max = 10)

        dnnregressor = tf.estimator.Estimator(
                                              model_fn = dnnreg_model_input.dnn_regressor,
                                              params = model_params,
                                              model_dir = basedir,
                                              config = config_obj
                                              warm_start_from = warm_start_ckpt_path
                                              )

        # Initialze epoch and training steps counters
        max_epoch = model_params['max_epoch']
        epoch_cnt = 0
        training_steps_cnt = 0

        # Initialize metrics
        old_metric = 1e10
        new_metric = 1e10 - 1

        # Training loop
        print('starting training................')

        while new_metric < old_metric and epoch_cnt < max_epoch:

            # Update reference metric
            old_metric = new_metric

            # Training the estimator
            dnnregressor.train(
                               input_fn = lambda: dnnreg_model_input.train_input_fn(features = self.train_features,
                                                           labels = self.train_labels,
                                                           batch_size = model_params['batch_size']),
                               steps = steps_until_eval
                              )

            # Evaluation Metrics
            evaluation_metrics_test = dnnregressor.evaluate(
                                                            input_fn = lambda: dnnreg_model_input.eval_input_fn(features = self.test_features,
                                                                                                                labels = self.test_labels,
                                                                                                                batch_size = 100,
                                                                                                                num_epochs = 1
                                                                                                                )
                                                            )

            # Update metric
            new_metric = evaluation_metrics_test[model_params['loss_fn']]
            training_steps_cnt += steps_until_eval

            # Print info about improvement of evaluation metric
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
                with open(basedir + '/dnn_training_results_' + model_params['loss_fn'] + self.hyper_params['train_test_file_signature'] + date_time_str + '.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)

            # Update training_results table with current training results
            with open(basedir + '/dnn_training_results_' + model_params['loss_fn'] + self.hyper_params['train_test_file_signature'] + date_time_str + '.csv', 'a') as f:
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
