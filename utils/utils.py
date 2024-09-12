#python libs
import re 
import math
from typing import Tuple
#import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision
#from tensorflow.keras import layers




def normalize(data, np_array=False, scaled=False):
    """
    Normalize and optionally scale a dataset.
    
    @param (DataFrame or numpy array) data: 2D DataFrame or numpy array
    @param (bool) np_array : optional np_array flag (default = false)
                             forces return type to numpy array
    @param (bool) scale : optionally scale the dataset (default = false)

    Return: Pandas DataFrame or Numpy array with normalized/scaled data
    """
    # ensure floats
    data = data.astype(float)
    
    if detect_datatype(data) == DataType.NUMPY:
        # set-up normalization
        high = 1.0
        low = 0.0
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        rng = maxs - mins
        # normalize
        data = high - (((high - low) * (maxs - data)) / rng)
        # scale if needed
        if scaled:
            data = (data / .5) - 1
    elif detect_datatype(data) == DataType.DATAFRAME:
        # normalize
        [data[col].update((data[col] - data[col].min()) / (data[col].max() - data[col].min())) for col in data.columns]
        # scale if needed
        if scaled:
            [data[col].update((data[col] / .5)-1) for col in data.columns]
    
    # return appropriate object
    if np_array and detect_datatype(data) == DataType.DATAFRAME:
        data = data.to_numpy()
    elif not np_array and detect_datatype(data) == DataType.NUMPY:
        data = pd.DataFrame(data)
    
    return data

def shape_3D_data(data, timesteps):
    if len(data.shape) != 2: raise TypeError("Input data must be 2D.")
    
    """
    Resape 2D data into 3D data of groups of 2D timesteps
    
    @param (DataFrame or numpy array) data: 2D DataFrame or numpy array
    @param (int) timesteps: number of timesteps/group

    Return: The reshaped data as numpy 3D array
    """ 

    # samples are total number of input vectors
    samples = data.shape[0]
    # time steps are steps per batch
    features = data.shape[1]
    
    # samples must divide evenly by timesteps to create an even set of batches
    if not(samples % timesteps):
        return np.array(data).reshape(int(data.shape[0] / timesteps), timesteps, features)
    else:
        msg = "timesteps must divide evenly into total samples: " + str(samples) + "/" \
            + str(timesteps) + "=" + str(round(float(samples) / float(timesteps), 2))
        raise ValueError(msg)



if __name__ == "__main__":
    # data = np.random.randn(79955, 3)
    # labels = np.random.randn(79955)
    # new_data  = sliding_window(data, labels, 99, 10034, 100, 100)
    # print(new_data.shape)
    #data, label = process_data(cfg.VALID, 128, 10)
    #np.savez('datasets/UniMb_val.npz', data= data, label = label)