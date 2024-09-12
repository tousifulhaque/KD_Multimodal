'''
Training Script
'''
import os
import sys
import shutil
from argparse import ArgumentParser
import yaml


sys.path.append('Kd_Multimodal/')
import  numpy as np
from numpy import argmax
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Add, Dense, LayerNormalization,GlobalAveragePooling1D
from tensorflow.keras.metrics import Recall, Precision
import matplotlib.pyplot as plt
#from sklearn.metrics import f1_score


#local imports

from utils.processing import process_data
from utils.imports import import_class
#config
def distribution_viz( labels, work_dir, mode):
    values, count = np.unique(labels, return_counts = True)
    plt.bar(x = values,data = count, height = count)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.savefig( work_dir + '/' + '{} Label Distribution'.format(mode))

def get_parser() -> ArgumentParser:
    '''
    Function to build a parser with argurments from config file
    '''
    parser = ArgumentParser(description='Arguements for experiment')
    parser.add_argument('--config',type = str, default='config/transformer.yaml')
    parser.add_argument('--model-path', type = str)
    parser.add_argument('--model-args', type = str)
    parser.add_argument('--dataset-args', type = str)
    parser.add_argument('--hyperparameters', type = str)
    parser.add_argument('--optimizer-args', type = str)
    parser.add_argument('--loss-args', type=str)
    parser.add_argument('--experiment-dir', type =str)
    parser.add_argument('--dataset', type = str)
    return parser



#creating model
if __name__ == '__main__':
    arg_parser = get_parser()
    p = arg_parser.parse_args()
    if p.config is not None: 
        ## save the config in the experiment here
        with open(file = p.config, mode='r',encoding='utf-8') as f: 
            default_arg = yaml.safe_load(f)
        
        key = vars(p).keys()
        for k in default_arg.keys():
            assert ( k in key)
            if k not in key:
                    raise ValueError(f'Argument {k} is out of scope')
        arg_parser.set_defaults(**default_arg)
    
    arg = arg_parser.parse_args()
    
    if not os.path.exists(arg.experiment_dir):
         os.makedirs(arg.experiment_dir)
         shutil.copy(arg.config, arg.experiment_dir)
    
    #dump config file

    model_class = import_class(arg.model_path)
    model = model_class(**arg.model_args)


    #processing train data 
    splited_dataset = process_data(**arg.dataset_args)

    #processing val data 
    X_train , y_train = splited_dataset['train']
    X_val, y_val = splited_dataset['val']
    X_test, y_test = splited_dataset['test']

    ## loss function 
    optimizer = Adam(**arg.optimizer_args)

    model.compile(
        loss= BinaryCrossentropy(label_smoothing=0.1),
        optimizer=Adam(
            **arg.optimizer_args
        ),
        metrics=[Recall(), Precision()],
        )
    checkpoint_filepath = os.path.join(os.getcwd(),
                                       f'{arg.experiment_dir}/model/{arg.dataset}_{arg.dataset_args["window"]}')
    model_checkpoint = ModelCheckpoint(filepath = checkpoint_filepath, 
                                        save_weights_only = False, 
                                        monitor = 'val_loss',
                                        mode = 'min', 
                                        save_best_only = True, 
                                        verbose = True)
    # log_dir = "logs/"  # Specify the directory where TensorBoard logs will be saved
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        X_train,
        y_train,
        **arg.hyperparameters,
        validation_data=(X_val, y_val),
        shuffle = True,
        callbacks=[
            #LearningRateScheduler(cosine_schedule(base_lr=config['learning_rate'], total_steps=config['epochs'], warmup_steps=config['warmup_steps'])),
            #EarlyStopping(monitor="loss", mode='min', min_delta=0.001, patience=5),
            model_checkpoint
        ],
        verbose=1
        )
