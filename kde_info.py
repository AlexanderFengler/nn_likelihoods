import numpy as np
import pickle
import cddm_data_simulation as cd
#import lba
import boundary_functions as bf
import os

temp = {"ddm":
    {
    "dgp": cd.ddm_flexbound,
    "boundary": bf.constant,
    "data_folder": "/users/afengler/data/kde/ddm/train_test_data_20000",
#     custom_objects: {"huber_loss": tf.losses.huber_loss}
#     fcn_path: "/users/afengler/data/tony/kde/ddm/keras_models/\
# deep_inference08_12_19_11_15_06/model.h5"
#    fcn_custom_objects: {"heteroscedastic_loss": tf.losses.huber_loss}
    "output_folder": "/users/afengler/data/kde/ddm/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm/method_comparison/",
    "param_names": ["v", "a", "w"],
    "boundary_param_names": [],
    "param_bounds": [[-2.0, 2.0], [0.6, 1.5], [0.3, 0.7]],   
    #"param_bounds": np.array([[-2, .6, .3], [2, 1.5, .7]]),
    "boundary_param_bounds": []
    },
"linear_collapse":
    {
    "dgp": cd.ddm_flexbound,
    "boundary": bf.linear_collapse,
    "data_folder": "/users/afengler/data/kde/linear_collapse/train_test_data_20000",
    "output_folder": "/users/afengler/data/kde/linear_collapse/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/linear_collapse/method_comparison/",
    "param_names": ["v", "a", "w"],
    "boundary_param_names": ["node", "theta"],
    "param_bounds": [[-2, 2], [0.6, 1.5], [0.3, 0.7]],
    "boundary_param_bounds": [[1, 2], [0, 1.37]]
    #"param_bounds": np.array([[-2, .6, .3], [2, 1.5, .7]]),
    #"boundary_param_bounds": np.array([[1, 0], [2, 1.37]])
    },
"ornstein":
    {
    "dgp": cd.ornstein_uhlenbeck,
    "boundary": bf.constant,
    "data_folder": "/users/afengler/data/kde/ornstein_uhlenbeck/train_test_data_20000",
    "output_folder": "/users/afengler/data/kde/ornstein_uhlenbeck/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ornstein_uhlenbeck/method_comparison/",
    "param_names": ["v", "a", "w", "g"],
    "boundary_param_names": [],
    "boundary_param_bounds": [],
    "param_bounds": [[-2, 2], [0.6, 1.5], [0.3, 0.7], [-1, 1]]
    #"param_bounds": np.array([[-2, .6, .3, -1], [2, 1.5, .7, 1]])
    },
"full":
    {
    "dgp": cd.full_ddm,
    "boundary": bf.constant,
    "data_folder": "/users/afengler/data/kde/full_ddm/train_test_data_20000",
    "output_folder": "/users/afengler/data/kde/full_ddm/method_comparison/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/full_ddm/method_comparison/",
    "param_names": ["v", "a", "w", "dw", "sdv"],
    "boundary_param_names": [],
    "boundary_param_bounds": [],
    "param_bounds": [[-2, 2], [0.6, 1.5], [0.3, 0.7], [0, 0.1], [0, 0.5]]   
    #"param_bounds": np.array([[-2, .6, .3, 0, 0], [2, 1.5, .7, .1, .5]])
    },
"ddm_fcn":
    {
    "dgp": cd.ddm_flexbound,
    "boundary": bf.constant,
    "data_folder": "/users/afengler/data/tony/kde/ddm/train_test_data_fcn",
    "output_folder": "/users/afengler/data/kde/ddm/method_comparison_fcn/",
    "output_folder_x7": "/media/data_cifs/afengler/data/kde/ddm/method_comparison_fcn/",
    "param_names": ["v", "a", "w"],
    "boundary_param_names": [],
    "param_bounds": [[-2, 2], [0.6, 1.5], [0.3, 0.7]],
    #"param_bounds": np.array([[-2, .6, .3], [2, 1.5, .7]]),
    "boundary_param_bounds": []
    }
}

pickle.dump(temp, open(os.getcwd() + "/kde_stats.pickle", "wb"))