import glob
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding
import re


def preprocessing():
    fall = os.path.join(os.getcwd() , 'datasets/Fall/***.xlsx')
    adl = os.path.join(os.getcwd(), 'datasets/ADL/***.xlsx')
    fall_files = glob.glob(fall)
    adl_files = glob.glob(adl)
    all_file_path = fall_files + adl_files
    trials = []
    labels = []
    fall_pattern = re.compile("Fall")
    for file_path in all_file_path:
        
        #checking if the excel has 2 sheets or not
        label = None
        if fall_pattern.search(file_path):
            label = 1
        else:
            label = 0
                    
        if len (pd.ExcelFile(file_path).sheet_names) == 2:
            df = pd.read_excel(fall_files[1], sheet_name=-1)
            df = df.iloc[:, :-1]
            null_col = df[df.isnull().any(axis = 1)].index.to_list()
            
            #calculating how many null segments we have 
            null_col = df[df.isnull().any(axis = 1)].index.to_list()
            null_seg = len(null_col)//10
            trial_start_lst = null_col[9::10]
            trial_end_lst = null_col[10::10]
            
            for i in range(len(null_col)//10 + 1):
                trial = None
                if i == 0 :
                    trial = df.iloc[0:null_col[1]-1, 3:6]
                elif i == null_seg :
                    trial = df.iloc[trial_start_lst[-1]+1:, 3:6]
                else: 
                    trial_end = trial_end_lst[i-1]
                    trial_start = trial_start_lst[i-1] + 1
                    trial = df.iloc[trial_start : trial_end-1 , 3:6].values
                    trial = tf.convert_to_tensor(trial, dtype = tf.float32)
                labels.append(label)
                trials.append(trial)
    trials = tf.keras.utils.pad_sequences(trials, maxlen= 512, value = float('-inf') , dtype = float, padding = 'post' )

    #transposing the trials 
    trials = tf.transpose(trials, perm = [0,2,1])

    try:
         np.savez_compressed("fall_detection_dataset", trials=trials, labels=labels)
         print('Creating Dataset successful')
    except:
        raise RuntimeError("Failed creating the dataset")
    

if __name__ == "__main__":
    preprocessing()