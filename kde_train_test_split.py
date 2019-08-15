# My own code
import kde_training_utilities as kde_utils

if __name__ == "__main__":
    # CHOOSE FOLDER
    # CCV 
    # ....
    # X7
    my_folder = '/media/data_cifs/afengler/data/kde/ornstein_uhlenbeck/train_test_data_20000/'
    kde_utils.kde_make_train_test_split(folder = my_folder, 
                                        p_train = 0.8)