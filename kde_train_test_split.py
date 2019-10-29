# My own code
import kde_training_utilities as kde_utils

if __name__ == "__main__":
    # CHOOSE FOLDER
    machine = 'x7'
    if machine == 'ccv':
        # CCV 
        my_folder = 'users/afengler/data/kde/weibull_cdf/train_test_data_ndt_20000/'
    if machine == 'x7':
        # X7
        my_folder = '/media/data_cifs/afengler/data/kde/weibull_cdf/train_test_data_ndt_20000/'
    print('Folder used:', my_folder)
    
    kde_utils.kde_make_train_test_split(folder = my_folder, 
                                        p_train = 0.99)