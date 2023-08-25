#python libs
import re 
import math
#import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision
#from tensorflow.keras import layers

#local libs
import cfg

def cosine_schedule(base_lr, total_steps, warmup_steps ):
    def step_fn(epoch):
        lr = base_lr 
        progress = (epoch - warmup_steps) / float(total_steps -  warmup_steps)

        progress = tf.clip_by_value(progress, 0.0, 1.0)

        lr = lr * 0.5 * (1.0 + tf.cos(math.pi * progress))
        
        if warmup_steps:
            lr = lr * tf.minimum(1.0 , epoch/warmup_steps)
        
        return lr
    

    return step_fn

class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = Precision(thresholds=0.5)
        self.recall_fn = Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)

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

def sliding_window(data,labels, clearing_time_index, max_time, sub_window_size, stride_size):

    assert clearing_time_index >= sub_window_size - 1 , "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 

    if max_time >= data.shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1
        # 2510 // 100 - 1 25 #25999 1000 24000 = 24900

    sub_windows  = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step = stride_size), 0).T
    )
    print(sub_windows.shape)
    labels = np.round(np.mean(labels[sub_windows], axis=1))
    return data[sub_windows], labels

def process_data(file_path, window_size, stride):
    dataframe = pd.read_csv(file_path)
    dataset = dataframe[['w_accelerometer_x', 'w_accelerometer_y', 'w_accelerometer_z']].to_numpy()
    labels = dataframe['outcome'].to_numpy()
    data, labels= sliding_window(dataset, labels, window_size - 1,dataset.shape[0],window_size,stride)
    return data, labels


if __name__ == "__main__":
    data = np.random.randn(79955, 3)
    labels = np.random.randn(79955)
    new_data  = sliding_window(data, labels, 99, 10034, 100, 100)
    print(new_data.shape)