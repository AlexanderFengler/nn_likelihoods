import os
import shutil

cwd = os.getcwd()

def cleaner(models = True,
            hyper_param_file = True,
            train_data = True,
            test_data = True,
            results = True,
            best_hyperparams = True):
    if models:
        #shutil.rmtree(cwd + '/tensorflow_models')
        print('ok')

    if hyper_param_file:
        os.remove(cwd + '/hyper_parameters.csv')
        print('ok')

    if train_data:
        os.remove(cwd + '/train_data*')
        print('ok')

    if test_data:
        os.remove(cwd + '/test_data*')
        print('ok')

    if results:
        os.remove(cwd + '/dnnregressor_result_table.csv')
        print('ok')

    if best_hyperparams:
        os.remove(cwd + '/best_hyperparams*')
        print('ok')
    return

if __name__ == "__main__":
    models = input('Delete models? (Y/N)')
    assert models == 'Y' or models =='N'
    hyper_param_file = input('Delete Hyperparameter csv file? (Y/N)')
    assert hyper_param_file == 'Y' or hyper_param_file =='N'
    train_data = input('Delete training data files? (Y/N)')
    assert train_data == 'Y' or train_data =='N'
    test_data = input('Delete test data files? (Y/N)')
    assert test_data == 'Y' or test_data =='N'
    results = input('Delete training results csv? (Y/N)')
    assert results == 'Y' or results =='N'
    best_hyperparams = input('Delete best Hyperparameter csv? (Y/N)')
    assert best_hyperparams == 'Y' or best_hyperparams =='N'

    if models == 'Y':
        models = True

    if hyper_param_file == 'Y':
        hyper_param_file = True

    if train_data == 'Y':
        train_data = True

    if test_data == 'Y':
        test_data = True

    if results == 'Y':
        results = True

    if best_hyperparams == 'Y':
        best_hyperparams = True


    cleaner(models = models,
            hyper_param_file = hyper_param_file,
            train_data = train_data,
            test_data = test_data,
            results = results,
            best_hyperparams = best_hyperparams
            )
