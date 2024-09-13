import shutil
import glob
import os
import re
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd

def get_arg():
    parser = argparse.ArgumentParser(description='Arguments for file transfer')
    parser.add_argument('--source', type = str, help = 'Source folder')
    parser.add_argument('--dest', type= str, help = 'Destination folder')
    parser.add_argument('--types', type= str, help= 'ADL/Fall')
    parser.add_argument('--data-type', type = str)
    args = parser.parse_args()
    return args


class ActivityFile:
    '''
    Class to store file meta data and processing data
    '''
    def __init__(self,file_path:str, directory:str):
        self.path = os.path.join(directory,file_path)
        self.subject_id = int(file_path[1:3])
        self.actvity_id = int(file_path[4:6])
    
    def read_data(self) -> Tuple[np.array, np.array]:
        '''
        Function to read data and label
        Out: 
            data : processed data
            label: labels for every time step
        '''
        dataframe = pd.read_csv(self.path)
        sequence_data = dataframe.to_numpy()[:, -3:]
        label = int(self.actvity_id>9)
        sequence_labels = np.repeat(label,sequence_data.shape[0])
        return sequence_data, sequence_labels
    
def find_match_elements(pattern, elements): 
    #compile the regular expression
    try:
        regex_pattern = re.compile(pattern)
        #filtering the elements that match the regex pattern
        matching_elements = [element for element in elements if regex_pattern.search(element)]
        return matching_elements
    except:
        print(f'Error: {e}')
        
    return []

def move_all(file_paths : List, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for path in file_paths:
        shutil.move(path, dest_folder)
    
def move_files(source_folder, destination_folder, pattern):
    try:
        
        # Check if the source folder exists
        if not os.path.exists(source_folder):
            raise FileNotFoundError("Source folder does not exist.")
        
        # Check if the destination folder exists, if not, create it
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Get a list of files in the source folder
        files = os.listdir(source_folder)
        matched_files = find_match_elements(pattern, files)
        
        if not matched_files:
            raise Exception('Couldn\'t find files with the pattern')

        
        for file in matched_files:
            
            source_file_path = os.path.join(source_folder, file)
            destination_file_path = os.path.join(destination_folder, file)

            # Perform the move operation
            shutil.move(source_file_path, destination_file_path)
        print("Files moved successfully.")
        
    except Exception as e:
        print(f"Error: {e}")


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
    labels = np.round(np.mean(labels[sub_windows], axis=1))
    return data[sub_windows], labels

def find_csv(directory: str) -> List[str]:
    filenames = os.listdir(directory)
    csv_files = [ActivityFile(f, directory) for f in filenames if f.endswith('.csv')]
    return csv_files

def segment_dataset(sequence_data : np.ndarray, sequence_labels: np.ndarray, 
                    window:int, stride:int) -> Tuple[np.ndarray,np.ndarray]:
    '''
    Function to segment data
    '''
    dataset = np.concatenate(sequence_data, axis=0)
    labels = np.concatenate(sequence_labels, axis = 0)
    assert dataset.shape[0] == labels.shape[0]
    segmented_dataset, segmented_labels= sliding_window(dataset, labels, window - 1,dataset.shape[0],window,stride)
    return segmented_dataset.astype(np.float32), segmented_labels.astype(np.float32)


def process_data(directory : str, train_subjects : str, val_subjects : List[int], 
                 window : int, stride : int) -> Tuple[np.ndarray, np.ndarray] :
    '''
    Data Processing 
    Args: 

    Outputs: 

    '''
    train_data = []
    train_label = []

    val_data  = []
    val_label = []

    test_data = []
    test_label = []

    activity_list = find_csv(directory)

    for activity in activity_list:
        trial_data, trial_label = activity.read_data()
        assert trial_data.shape[0] == trial_label.shape[0]
        if activity.subject_id in train_subjects:
            train_data.append(trial_data)
            train_label.append(trial_label)
        elif activity.subject_id in val_subjects:
            val_data.append(trial_data)
            val_label.append(trial_label)
        else:
            test_data.append(trial_data)
            test_label.append(trial_label)

    train_dataset, train_labels = segment_dataset(train_data, train_label, window, stride)
    val_dataset, val_labels = segment_dataset(val_data, val_label, window, stride)
    test_dataset, test_labels = segment_dataset(test_data, test_label, window, stride)
    splited_dataset = {
        'train' : (train_dataset, train_labels),
        'val' : (val_dataset, val_labels),
        'test' : (test_dataset, test_labels)

    }

    return splited_dataset


    

    


    
if __name__ == "__main__" :
    directory  = '/Users/tousif/Lstm_transformer/KD_Multimodal/datasets/smartfallmm/student_participants/accelerometer/watch_accelerometer'
    train_subjects = [29,31,32,36,37,40, 41, 42, 43,44,45,46]
    val_subjects = [34]
    window = 256
    stride = 10

    dataset = splited_dataset = process_data(directory, train_subjects, val_subjects, window, stride)
    X_train, y_train = dataset['train']
    print(X_train.shape)
    X_val, y_val = dataset['val']
    print(X_val.shape)
    X_test, y_test = dataset['test']
    print(X_test.shape)
    np.savez(os.path.join(os.getcwd(), 'datasets/smartfallmm/sm_train_256.npz'), data= X_train, label = y_train)
    np.savez(os.path.join(os.getcwd(),'datasets/smartfallmm/sm_val_256.npz'), data= X_val, label = y_val)
    np.savez(os.path.join(os.getcwd(), 'datasets/smartfallmm/sm_test_256.npz'), data= X_test, label = y_test)