import numpy as np
import pickle
import cddm_data_simulation as cd
import clba
import boundary_functions as bf
import os

temp = {
"test":{
    "dgp": cd.test,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/test/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/test/",
    "data_folder": "/users/afengler/data/kde/test/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/test/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/test/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/test/fcn_train_test_data_2000",
    #"custom_objects": {"huber_loss": tf.losses.huber_loss},
    #"fcn_custom_objects": {"heteroscedastic_loss": tf.losses.huber_loss},
    "output_folder": "/users/afengler/data/kde/test/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/test/method_comparison/",
    "model_folder": "/users/afengler/data/kde/test/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/test/keras_models/",
    "param_names": ['v', 'a', 'w', 'ndt'],
    "boundary_param_names": [],
    #"param_bounds_network": [[-2.0, 2.0], [0.5, 1.5], [0.3, 0.7], [0.0, 1.0]],  
    "param_bounds_network": [[-2.0, 2.0], [0.3, 2], [0.2, 0.8], [0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [0.2, 2], [0.1, 0.9], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn":[],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"lba":{
    "dgp": clba.rlba,
    "method_folder": '/users/afengler/data/kde/lba/',
    "method_folder_x7": '/media/data_cifs/afengler/data/kde/lba/',
    "data_folder": "/users/afengler/data/kde/lba/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/lba/train_test_data_ndt_20000/",
    "output_folder": "/users/afengler/data/kde/lba/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/lba/method_comparison/",
    "model_folder": "/users/afengler/data/kde/lba/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/lba/keras_models/",
    "param_names": ['v', 'A', 'b', 's', 'ndt'],
    "param_depends_on_n_choice": [1, 0, 0, 0, 0],
    "boundary_param_names": [],
    "param_bounds_network": [[1.0, 2.0], [1.0, 2.0], [0.0, 1.0], [1.5, 3.0], [0.1, 0.2], [0.0, 1.0]],
    "param_bounds_sampler": [[1.25, 1.75], [1.25, 1.75], [0.2, 0.8], [1.75, 2.75], [0.11, 0.19], [0.1, 0.9]], 
    "param_bounds_cnn": [[1.0, 2.0], [1.0, 2.0], [0.0, 1.0], [1.5, 3.0], [0.1, 0.2], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn": [],
    "dgp_hyperparameters": [['n_samples', 20000], 
                            ['max_t', 20.0], 
                            ['d_lower_lim', 0.01]],
       },
"lba_analytic":{
    "dgp": clba.rlba,
    "method_folder": '/users/afengler/data/analytic/lba/',
    "method_folder_x7": '/media/data_cifs/afengler/data/analytic/lba/',
    "data_folder": "/users/afengler/data/analytic/lba/train_test_data_kde_imit/",
    "data_folder_x7": "/media/data_cifs/afengler/data/analytic/lba/train_test_data_kde_imit/",
    "output_folder": "/users/afengler/data/analytic/lba/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/analytic/lba/method_comparison/",
    "model_folder": "/users/afengler/data/analytic/lba/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/analytic/lba/keras_models/",
    "param_names": ['v', 'A', 'b', 's', 'ndt'],
    "param_depends_on_n_choice": [1, 0, 0, 0, 0],
    "boundary_param_names": [],
    "param_bounds_network": [[1.0, 2.0], [1.0, 2.0], [0.0, 1.0], [1.5, 3.0], [0.1, 0.2], [0.0, 1.0]],
    "param_bounds_sampler": [[1.25, 1.75], [1.25, 1.75], [0.2, 0.8], [1.75, 2.75], [0.11, 0.19], [0.1, 0.9]], 
    "param_bounds_cnn": [[1.0, 2.0], [1.0, 2.0], [0.0, 1.0], [1.5, 3.0], [0.1, 0.2], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "param_bounds_cnn": [],
    "dgp_hyperparameters": [['n_samples', 20000],
                            ['max_t', 20.0], 
                            ['d_lower_lim', 0.01]],
    },
"levy":{
    "dgp": cd.levy_flexbound,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/levy/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/levy/",
    "data_folder": "/users/afengler/data/kde/levy/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/levy/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/levy/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/levy/fcn_train_test_data_2000",
    #"custom_objects": {"huber_loss": tf.losses.huber_loss},
    #"fcn_custom_objects": {"heteroscedastic_loss": tf.losses.huber_loss},
    "output_folder": "/users/afengler/data/kde/levy/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/levy/method_comparison/",
    "model_folder": "/users/afengler/data/kde/levy/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/levy/keras_models/",
    "param_names": ['v', 'a', 'w', 'alpha_diff', 'ndt'],
    "boundary_param_names": [],
    #"param_bounds_network": [[-2.0, 2.0], [0.5, 1.5], [0.3, 0.7], [0.0, 1.0]],  
    "param_bounds_network": [[-2.0, 2.0], [0.3, 2], [0.2, 0.8], [1.0, 2], [0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.4, 2], [0.3, 0.7], [1.1, 1.9], [0.1, 1.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [0.2, 2], [0.1, 0.9], [1.0, 2.0], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn":[],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"ddm":{
    "dgp": cd.ddm_flexbound,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/ddm/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/ddm/",
    "data_folder": "/users/afengler/data/kde/ddm/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/ddm/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/ddm/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/ddm/fcn_train_test_data_2000",
    #"custom_objects": {"huber_loss": tf.losses.huber_loss},
    #"fcn_custom_objects": {"heteroscedastic_loss": tf.losses.huber_loss},
    "output_folder": "/users/afengler/data/kde/ddm/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm/method_comparison/",
    "model_folder": "/users/afengler/data/kde/ddm/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/ddm/keras_models/",
    "param_names": ['v', 'a', 'w', 'ndt'],
    "boundary_param_names": [],
    #"param_bounds_network": [[-2.0, 2.0], [0.5, 1.5], [0.3, 0.7], [0.0, 1.0]],  
    "param_bounds_network": [[-2.0, 2.0], [0.3, 2], [0.2, 0.8], [0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [0.2, 2], [0.1, 0.9], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn":[],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"ddm_seq2":{
    "dgp": cd.ddm_flexbound_seq2,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/ddm_seq2/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_seq2/",
    "data_folder": "/users/afengler/data/kde/ddm_seq2/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_seq2/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/ddm_seq2/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/ddm_seq2/fcn_train_test_data_2000",
    "output_folder": "/users/afengler/data/kde/ddm_seq2/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_seq2/method_comparison/",
    "model_folder": "/users/afengler/data/kde/ddm_seq2/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_seq2/keras_models/",
    "param_names": ['v_h', 'v_l_1', 'v_l_2', 'a', 'w_h', 'w_l_1', 'w_l_2', 'ndt'],
    "param_depends_on_n_choice": [0, 0, 0, 0, 0, 0, 0, 0],
    "boundary_param_names": [],
    "param_bounds_network": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                             [0.3, 2], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8],[0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                         [0.2, 2], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn":[],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"ddm_seq2_angle":{
    "dgp": cd.ddm_flexbound_seq2,
    "boundary": bf.angle,
    "boundary_multiplicative": False,
    "method_folder": "/users/afengler/data/kde/ddm_seq2_angle/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_seq2_angle/",
    "data_folder": "/users/afengler/data/kde/ddm_seq2_angle/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_seq2_angle/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/ddm_seq2_angle/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/ddm_seq2_angle/fcn_train_test_data_2000",
    "output_folder": "/users/afengler/data/kde/ddm_seq2_angle/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_seq2_angle/method_comparison/",
    "model_folder": "/users/afengler/data/kde/ddm_seq2_angle/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_seq2_angle/keras_models/",
    "param_names": ['v_h', 'v_l_1', 'v_l_2', 'a', 'w_h', 'w_l_1', 'w_l_2', 'ndt'],
    "param_depends_on_n_choice": [0, 0, 0, 0, 0, 0, 0, 0],
    "boundary_param_names": ['theta'],
    "param_bounds_network": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                             [0.3, 2], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8],[0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                         [0.2, 2], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.0, 2.0]],
    "boundary_param_bounds_network":[[0, (np.pi / 2 - .1)]],
    "boundary_param_bounds_sampler": [[0.05, np.pi / 2 - .3]],
    "boundary_param_bounds_cnn": [[0, (np.pi / 2 - .2)]],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.angle],
                            ['boundary_multiplicative', False]],
    },
"ddm_par2":{
    "dgp": cd.ddm_flexbound_par2,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/ddm_par2/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_par2/",
    "data_folder": "/users/afengler/data/kde/ddm_par2/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_par2/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/ddm_par2/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/ddm_par2/fcn_train_test_data_2000",
    "output_folder": "/users/afengler/data/kde/ddm_par2/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_par2/method_comparison/",
    "model_folder": "/users/afengler/data/kde/ddm_par2/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_par2/keras_models/",
    "param_names": ['v_h', 'v_l_1', 'v_l_2', 'a', 'w_h', 'w_l_1', 'w_l_2', 'ndt'],
    "param_depends_on_n_choice": [0, 0, 0, 0, 0, 0, 0, 0],
    "boundary_param_names": [],
    "param_bounds_network": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                             [0.3, 2], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8],[0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                         [0.2, 2], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn":[],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"ddm_par2_angle":{
    "dgp": cd.ddm_flexbound_par2,
    "boundary": bf.angle,
    "boundary_multiplicative": False,
    "method_folder": "/users/afengler/data/kde/ddm_par2_angle/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_par2_angle/",
    "data_folder": "/users/afengler/data/kde/ddm_par2_angle/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_par2_angle/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/ddm_par2_angle/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/ddm_par2_angle/fcn_train_test_data_2000",
    "output_folder": "/users/afengler/data/kde/ddm_par2_angle/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_par2_angle/method_comparison/",
    "model_folder": "/users/afengler/data/kde/ddm_par2_angle/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_par2_angle/keras_models/",
    "param_names": ['v_h', 'v_l_1', 'v_l_2', 'a', 'w_h', 'w_l_1', 'w_l_2', 'ndt'],
    "param_depends_on_n_choice": [0, 0, 0, 0, 0, 0, 0, 0],
    "boundary_param_names": ['theta'],
    "param_bounds_network": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                             [0.3, 2], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8],[0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                         [0.2, 2], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.0, 2.0]],
    "boundary_param_bounds_network":[[0, (np.pi / 2 - .1)]],
    "boundary_param_bounds_sampler": [[0.05, np.pi / 2 - .3]],
    "boundary_param_bounds_cnn": [[0, (np.pi / 2 - .2)]],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.angle],
                            ['boundary_multiplicative', False]],
    },
"ddm_mic2":{
    "dgp": cd.ddm_flexbound_mic2,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/ddm_mic2/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_mic2/",
    "data_folder": "/users/afengler/data/kde/ddm_mic2/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_mic2/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/ddm_mic2/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/ddm_mic2/fcn_train_test_data_2000",
    "output_folder": "/users/afengler/data/kde/ddm_mic2/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_mic2/method_comparison/",
    "model_folder": "/users/afengler/data/kde/ddm_mic2/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_mic2/keras_models/",
    "param_names": ['v_h', 'v_l_1', 'v_l_2', 'a', 'w_h', 'w_l_1', 'w_l_2', 'd' ,'ndt'],
    "param_depends_on_n_choice": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "boundary_param_names": [],
    "param_bounds_network": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                             [0.3, 2], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8], [0.0, 1.0], [0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                            [0.2, 2], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.0, 1.0], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn":[],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"ddm_mic2_angle":{
    "dgp": cd.ddm_flexbound_mic2,
    "boundary": bf.angle,
    "boundary_multiplicative": False,
    "method_folder": "/users/afengler/data/kde/ddm_mic2_angle/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_mic2_angle/",
    "data_folder": "/users/afengler/data/kde/ddm_mic2_angle/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_mic2_angle/train_test_data_ndt_20000/",
    "data_folder_fcn": "/users/afengler/data/analytic/ddm_mic2_angle/fcn_train_test_data_2000",
    "data_folder_fcn_x7": "/media/data_cifs/afengler/data/analytic/ddm_mic2_angle/fcn_train_test_data_2000",
    "output_folder": "/users/afengler/data/kde/ddm_mic2_angle/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_mic2_angle/method_comparison/",
    "model_folder": "/users/afengler/data/kde/ddm_mic2_angle/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/ddm_mic2_angle/keras_models/",
    "param_names": ['v_h', 'v_l_1', 'v_l_2', 'a', 'w_h', 'w_l_1', 'w_l_2', 'd' ,'ndt'],
    "param_depends_on_n_choice": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "boundary_param_names": ['theta'],
    "param_bounds_network": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                             [0.3, 2], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8], [0.0, 1.0], [0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                            [0.2, 2], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.0, 1.0], [0.0, 2.0]],
    "boundary_param_bounds_network":[[0, (np.pi / 2 - .1)]],
    "boundary_param_bounds_sampler": [[0.05, np.pi / 2 - .3]],
    "boundary_param_bounds_cnn": [[0, (np.pi / 2 - .2)]],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.angle],
                            ['boundary_multiplicative', False]],
    },
"ddm_analytic":{
    "dgp": cd.ddm_flexbound,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder":"/users/afengler/data/analytic/ddm/",
    "method_folder_x7": "/media/data_cifs/afengler/data/analytic/ddm/",
    "data_folder": "/users/afengler/data/analytic/ddm/train_test_data_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/analytic/ddm/train_test_data_20000/",
    "output_folder": "/users/afengler/data/analytic/ddm/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/analytic/ddm/method_comparison/",
    "param_names": ['v', 'a', 'w', 'ndt'],
    "boundary_param_names": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_cnn": [],
    # "param_bounds_network": [[-2.0, 2.0], [0.5, 1.5], [0.3, 0.7], [0.0, 1.0]],
    "param_bounds_network": [[-2.0, 2.0], [0.3, 2], [0.2, 0.8], [0.0, 2.0]],
    "param_bounds_sampler": [[-2.0, 2.0], [0.6, 1.5], [0.30, 0.70], [0.0, 1.0]],
    "param_bounds_cnn": [[-2.0, 2.0], [0.2, 2], [0.1, 0.0], [0.0, 2.0]],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"angle":{
    "dgp": cd.ddm_flexbound,
    "boundary": bf.angle,
    "boundary_multiplicative": False,
    "method_folder": "/users/afengler/data/kde/angle/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/angle/",
    "data_folder": "/users/afengler/data/kde/angle/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/angle/train_test_data_ndt_20000/",
    "output_folder": "/users/afengler/data/kde/angle/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/angle/method_comparison/",
    "model_folder": "/users/afengler/data/kde/angle/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/angle/keras_models/",
    "param_names": ["v", "a", "w", "ndt"],
    "boundary_param_names": ["theta"],
    # "param_bounds_network": [[-1.5, 1.5], [0.6, 1.5], [0.3, 0.7], [0.0, 1.0]],
    "param_bounds_network": [[-2.0, 2.0], [0.3, 2], [0.2, 0.8], [0.0, 2.0]],
    "param_bounds_sampler": [[-1.51, 1.49], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [0.2, 2.0], [0.1, 0.9], [0.0, 2.0]],
    #"boundary_param_bounds_network": [[0, (np.pi / 2 - .2)]],
    'boundary_param_bounds_network':[[0, (np.pi / 2 - .1)]],
    "boundary_param_bounds_sampler": [[0.05, np.pi / 2 - .3]],
    "boundary_param_bounds_cnn": [[0, (np.pi / 2 - .2)]],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.angle],
                            ['boundary_multiplicative', False]],
    },
"weibull_cdf":{
    "dgp": cd.ddm_flexbound,
    "boundary": bf.weibull_cdf,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/weibull_cdf/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/weibull_cdf/",
    "data_folder": "/users/afengler/data/kde/weibull_cdf/train_test_data_ndt_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/weibull_cdf/train_test_data_ndt_20000/",
    "output_folder": "/users/afengler/data/kde/weibull_cdf/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/weibull_cdf/method_comparison/",
    "model_folder": "/users/afengler/data/kde/weibull_cdf/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/weibull_cdf/keras_models/",
    "param_names": ["v", "a", "w", "ndt"],
    "boundary_param_names": ["alpha", "beta"],
    # "param_bounds_network": [[-1.5, 1.5], [0.6, 1.5], [0.3, 0.7], [0.0, 1.0]],
    "param_bounds_network": [[-2.0, 2.0], [0.3, 2], [0.2, 0.8], [0.0, 2.0]],
    "param_bounds_sampler": [[-1.51, 1.49], [0.6, 1.4], [0.31, 0.69], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [0.2, 2.0], [0.1, 0.9], [0.0, 2.0]],
    #"boundary_param_bounds_network": [[0.5, 5.0], [0.5, 7.0]],
    "boundary_param_bounds_network": [[0.3, 5.0], [0.3, 7.0]],
    "boundary_param_bounds_sampler": [[0.55, 4.95], [0.55, 6.95]],
    "boundary_param_bounds_cnn": [[0.5, 5.0], [0.5, 7.0]],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.weibull_cdf],
                            ['boundary_multiplicative', True]],
    },
"ornstein":{
    "dgp": cd.ornstein_uhlenbeck,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/ornstein/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/ornstein/",
    "data_folder": "/users/afengler/data/kde/ornstein/train_test_data_20000/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/ornstein/train_test_data_20000/",
    "output_folder": "/users/afengler/data/kde/ornstein/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ornstein/method_comparison/",
    "model_folder": "/users/afengler/data/kde/orstein/keras_models/",
    "model_folder_x7": "/media/data_cifs/afengler/data/kde/ornstein/keras_models/",
    "param_names": ["v", "a", "w", "g", "ndt"],
    # "param_bounds_network": [[- 1.5, 1.5], [0.5, 1.5], [0.3, 0.7], [- 1.0, 1.0], [0.0, 1.0]],
    'param_bounds_network': [[-2.0, 2.0], [0.3, 2], [0.2, 0.8], [-1.0, 1.0], [0.0, 2.0]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69], [-0.9, 0.9], [0.1, 0.9]],
    "param_bounds_cnn": [[-2.5, 2.5], [0.2, 2.0], [0.1, 0.9], [-1.0, 1.0], [0.0, 2.0]],
    "boundary_param_names": [],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn": [],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"ornstein_angle":{},
"ornstein_weibull":{},
"full_ddm":{
    "dgp": cd.full_ddm,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/full_ddm/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/full_ddm/",
    "data_folder": "/users/afengler/data/kde/full_ddm/train_test_data_20000",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/full_ddm/train_test_data_20000",
    "output_folder": "/users/afengler/data/kde/full_ddm/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/full_ddm/method_comparison/",
    "param_names": ["v", "a", "w", "ndt", "dw", "sdv", "dndt"],
    "boundary_param_names": [],
    # "param_bounds_network": [[-2.0, 2.0], [0.6, 1.8], [0.3, 0.7], [0.25, 1.25], [0, 0.4], [0, 0.5], [0.0, 0.5]],
    "param_bounds_network": [[-2.0, 2.0], [0.3, 2], [0.3, 0.7], [0.25, 2.25], [0, 0.4], [0, 0.5], [0.0, 0.5]],
    "param_bounds_sampler": [[-1.9, 1.9], [0.65, 1.75], [0.31, 0.69], [0.3, 1.2], [0.05, 0.35], [0.05, 0.45], [0.05, 0.45]],
    "param_bounds_cnn": [[-2.5, 2.5], [0.2, 2.0], [0.1, 0.9], [0.25, 2.5], [0, 0.2], [0, 1], [0.0, 0.5]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn": [],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01],
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"race_model":{
    "dgp": cd.race_model,
    "boundary": bf.constant, 
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/race_model/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/race_model/",
    "data_folder": "/users/afengler/data/kde/race_model/train_test_data/",
    "data_folder_x7": "/media/data_cifs/afengler/data/kde/race_model/train_test_data/",
    "output_folder": "/users/afengler/data/kde/race_model/method_comparison/",
    "output_folder_x7":"/media/data_cifs/afengler/data_kde/race_model/train_test_data/",
    "param_names": ["v", "a", "w", "ndt"],
    "param_depends_on_n_choice": [1, 0, 1, 0],
    "boundary_param_names": [],
    "param_bounds_network":[[0, 2.0], [1.0, 3.0], [0.2, 0.8], [0.0, 1.0]],
    "param_bounds_sampler": [[0.1, 1.9], [1.1, 2.9], [0.21, 0.79], [0.1, 0.9]],
    "param_bounds_cnn": [[0.0, 2.5], [1.0, 3.0], [0.1, 0.9], [0.0, 2.0]],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "boundary_param_bounds_cnn": [],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
    },
"lca":{
    "dgp": cd.lca,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "method_folder": "/users/afengler/data/kde/lca/",
    "method_folder_x7": "/media/data_cifs/afengler/data/kde/lca/",
    "data_folder": "users/afengler/data/kde/lca/.../",
    "data_folder_x7": "users/afengler/data/kde/lca/.../",
    "output_folder": "/users/afengler/data/kde/lca/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/lca/method_comparison/",
    "param_names": ['v', 'a', 'w', 'g', 'b', 'ndt'],
    "param_depends_on_n_choice": [1, 0, 1, 0, 0, 0],
    "boundary_param_names": [],
    "param_bounds_network": [[0, 2.0], [1.0, 3.0], [0.2, 0.8], [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
    "param_bounds_sampler": [[0, 2.0], [1.0, 3.0], [0.2, 0.8], [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
    "param_bounds_cnn": [[0, 2.5], [1.0, 3.0], [0.1, 0.9], [-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]],
    "boundary_param_bounds_cnn": [],
    "boundary_param_bounds_network": [],
    "boundary_param_bounds_sampler": [],
    "dgp_hyperparameters": [['s', 1.0], 
                            ['delta_t', 0.01], 
                            ['max_t', 20], 
                            ['n_samples', 20000], 
                            ['print_info', False],
                            ['boundary', bf.constant],
                            ['boundary_multiplicative', True]],
},
"ddm_fcn":{
    "dgp": cd.ddm_flexbound,
    "boundary": bf.constant,
    "boundary_multiplicative": True,
    "data_folder": "/users/afengler/data/tony/kde/ddm/train_test_data_fcn",
    "output_folder": "/users/afengler/data/kde/ddm/method_comparison_fcn/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm/method_comparison_fcn/",
    "param_names": ["v", "a", "w"],
    "boundary_param_names": [],
    "param_bounds": [[-2, 2], [0.6, 1.5], [0.3, 0.7]],
    "boundary_param_bounds": []
    }
}

pickle.dump(temp, open(os.getcwd() + "/kde_stats.pickle", "wb"))

# "ddm":{
#     "dgp": cd.ddm_flexbound,
#     "boundary": bf.constant,
#     "boundary_multiplicative": True,
#     "data_folder": "/users/afengler/data/kde/ddm/train_test_data_20000",
# #     custom_objects: {"huber_loss": tf.losses.huber_loss}
# #     fcn_path: "/users/afengler/data/tony/kde/ddm/keras_models/\
# # deep_inference08_12_19_11_15_06/model.h5"
# #    fcn_custom_objects: {"heteroscedastic_loss": tf.losses.huber_loss}
#     "output_folder": "/users/afengler/data/kde/ddm/method_comparison/",
#     "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm/method_comparison/",
#     "param_names": ['v', 'a', 'w'],
#     "boundary_param_names": [],
#     "param_bounds_network": [[-2.0, 2.0], [0.5, 1.5], [0.3, 0.7]],
#     "param_bounds": [[-1.9, 1.9], [0.6, 1.4], [0.31, 0.69]],
#     "boundary_param_bounds": []
#     },

# "lba":{
#         "dgp":clba.rlba,
#         "data_folder": "/users/afengler/data/kde/lba/train_test_data_20000",
#         "data_folder_x7": "/media/data_cifs/afengler/data/kde/lba/train_test_data_20000",
#         "output_folder": "/users/afengler/data/kde/lba/method_comparison/",
#         "output_folder_x7": "/media/data_cifs/afengler/data/kde/lba/method_comparison/",
#         "model_folder": "/users/afengler/data/kde/lba/keras_models/",
#         "model_folder_x7": "/media/data_cifs/afengler/data/kde/lba/keras_models/",
#         "param_names": ['v_0', 'v_1', 'A', 'b', 's'],
#         "boundary_param_names": [],
#         "param_bounds_network": [[1.0, 2.0], [1.0, 2.0], [0.0, 1.0], [1.5, 3.0], [0.1, 0.2]],
#         "param_bounds_sampler": [[1.25, 1.75], [1.25, 1.75], [0.2, 0.8], [1.75, 2.75], [0.11, 0.19], [0.1, 0.9]], 
#         "boundary_param_bounds": []
#        },