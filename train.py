import os
from functools import * 
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import os
import logging



#from matplotlib import pyplot
import numpy as np
from numpy import argmax

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
from cfg import config, TRAIN, VALID, DATASET, WINDOW, STRIDE
from transformer import transformer
from utils import process_data , cosine_schedule, linear_scheduler

import warnings
import logging
import os
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
#config
def distribution_viz( labels, work_dir, mode):
    values, count = np.unique(labels, return_counts = True)
    plt.bar(x = values,data = count, height = count)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.savefig( work_dir + '/' + '{} Label Distribution'.format(mode))

#creating model
if __name__ == '__main__':
    #tf.debugging.set_log_device_placement(True)
    with tf.device('/device:GPU:0'):
        saved_dir = f'tmp/{DATASET}/{WINDOW}/'
        # strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        
        train_data= np.load(TRAIN)
        X_train, y_train = train_data['data'], train_data['labels'].astype(np.int64)
        #X_train = np.reshape(X_train, (-1,config['channel'],config['length']))
        #distribution_viz(y_train, saved_dir, "Train")

        #processing val data 
        val_data= np.load(VALID)
        X_val, y_val = val_data['data'], val_data['labels'].astype(np.int64)
        #X_val = np.reshape(X_val, (-1,config['channel'],config['length']))
        distribution_viz(y_val, saved_dir, "Validation")

        # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        # val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # train_data = train_dataset.with_options(options).batch(config['batch_size']*strategy.num_replicas_in_sync).prefetch(64)
        # val_data = val_dataset.with_options(options).batch(config['batch_size']*strategy.num_replicas_in_sync).prefetch(64)

        #learning rate scheduler 
        lr_fn = partial(linear_scheduler, dim_embed = config['embed_layer_size'])
        

        model = transformer(length = config['length'],
            channels=config['channel'],
            num_heads=config['num_heads'],
            dropout_rate = config['dropout'],
            attention_dropout_rate = config['attention_dropout'],
            embed_dim =config['embed_layer_size'],
            mlp_dim = config['fc_layer_size'],
            num_layers = config['num_layers'])


        #processing train data 


        model.compile(
            loss= BinaryCrossentropy(),
            optimizer=Adam(learning_rate = 0.001),
            metrics=[Recall(), Precision(),'accuracy'],
            )
        
        checkpoint_filepath = os.path.join(os.getcwd(), saved_dir+"{epoch:02d}_{val_accuracy:.2f}.hdf5")
        model_checkpoint = ModelCheckpoint(filepath = checkpoint_filepath, 
                                            save_weights_only = True, 
                                            monitor = 'val_accuracy',
                                            mode = 'max', 
                                            save_best_only = True, 
                                            verbose = True)

        #log_dir = "logs/"  # Specify the directory where TensorBoard logs will be saved
        model.summary()
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(
            X_train,
            y_train,
            epochs=config['epochs'],
            validation_data=(X_val, y_val),
            shuffle = True,
            callbacks=[
                # LearningRateScheduler(lr_fn(base_lr=config['learning_rate'], total_steps=config['epochs'], warmup_steps=config['warmup_steps']), verbose = 1),
                EarlyStopping(monitor="loss", mode='min', min_delta=0.001, patience=5, restore_best_weights = True),
                model_checkpoint
            ],
            verbose=1
            )
