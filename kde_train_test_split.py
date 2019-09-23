# My own code
import kde_training_utilities as kde_utils

if __name__ == "__main__":
    # CHOOSE FOLDER
    machine = 'ccv'
    if machine == 'ccv':
        # CCV 
        my_folder = 'users/afengler/data/kde/lba/train_test_data_20000/
    if machine == 'x7':
        # X7
        #my_folder = '/media/data_cifs/afengler/data/lba_analytic/train_test_data_kde_imit/'
    kde_utils.kde_make_train_test_split(folder = my_folder, 
                                        p_train = 0.99)