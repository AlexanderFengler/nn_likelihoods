import pickle
import numpy as np
import os
import argparse

if __name__ == "__main__":
    
    # Interface ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--datafolder",
                     type = str,
                     default = 'none')
    CLI.add_argument("--newdatafolder",
                     type = str,
                     default = 'none')
    
    
    files_ = os.listdir(datafolder)
    files_ = files[:10]
    # preprocess
    tmp = pickle.load(open(datafolder + file_[0], 'rb'))
    nchoices = len(np.unique(tmp[:, -2]))
    choices_sorted = np.unique(tmp[:, -2])
    choices_sorted.sort()
    new_data = np.zeros((tmp.shape[0], tmp.shape[1] + nchoices - 1))
    
    if not os.path.exists(newdatafolder):
        os.makedirs(newdatafolder)
    
    for file_ in files_:
        print('processing file: ', file_)
        data = pickle.load(open(datafolder + file_, 'rb'))
        new_data[:, : -(nchoices + 1)] = data[:, :(-2)]
        
        for choice_cnt in range(nchoices):
            new_data[:, - (nchoices + 1 - choice_cnt)] = (data[:, -2] == choices_sorted[choice_cnt]).astype(np.int)
    
        
        print('writing to new file: ', newdatafolder + file_)
        pickle.dump(new_data, open(newdatafolder + file_, 'wb'))
        
        
    
        
            
    


                     
                     
                     
                     