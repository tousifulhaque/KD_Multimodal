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
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Add, Dense, LayerNormalization,GlobalAveragePooling1D
from tensorflow.keras.metrics import Recall, Precision
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
#from sklearn.metrics import f1_score


#local imports
from cfg import config, TRAIN, VALID,TEST, DATASET, WINDOW, STRIDE
from transformer import transformer
from loss import WeightedBinaryCrossentropy
from utils import process_data , cosine_schedule, linear_scheduler, SaveLossCurve, F1_Score

import warnings
import logging
import os
import argparse 

warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def get_parser():
    arg_parser = argparse.ArgumentParser(description = 'Arguements for training')
    arg_parser.add_argument('--work-dir', type = str, help = 'Working Directory')
    return arg_parser

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
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    else : 
        print('Folder {} already exists'.format(args.work_dir))

    work_dir = args.work_dir

    with tf.device('/device:GPU:4'):
        # strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        
        # train_zip= np.load(TRAIN)
        # data, labels = train_zip['data'], train_zip['labels']
        data, labels = process_data(TRAIN, config['length'], 10)
        X_train, y_train = data, labels.astype(np.int64)
        neg, pos = np.bincount(y_train)
        total = y_train.shape[0]
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        initial_bias = np.log([pos/neg])

        #sample_weight = compute_sample_weight( y = y_train, indices=None)
        #print(sample_weight)
        # X_train, y_train = train_data['data'], train_data['labels'].astype(np.int64)
        # X_train = np.reshape(X_train, (-1,config['channel'],config['length']))
        distribution_viz(y_train, work_dir, "Train")

        #processing val data 
        # val_zip= np.load(VALID)
        # val_data, val_labels = val_zip['data'], val_zip['labels']
        val_data, val_labels = process_data(VALID, config['length'], 10)
        X_val, y_val = val_data, val_labels.astype(np.int64)
        #X_val, y_val = val_data['data'], val_data['labels'].astype(np.int64)
        #X_val = np.reshape(X_val, (-1,config['channel'],config['length']))
        distribution_viz(y_val, work_dir, "Validation")

        # test_zip = np.load(TEST)
        # test_data, test_labels = test_zip['data'], test_zip['labels']
        # X_test, y_test = test_data['data'], test_data['labels'].astype(np.int64)
        test_data, test_labels = process_data(TEST, config['length'], 10)
        X_test, y_test = test_data, test_labels.astype(np.int64)
        # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        # train_dataset = train_dataset.batch(batch_size = config['batch_size'])
        # val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        # val_dataset = val_dataset.batch(batch_size = config['batch_size'])

        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # train_data = train_dataset.with_options(options).batch(config['batch_size']*strategy.num_replicas_in_sync).prefetch(64)
        # val_data = val_dataset.with_options(options).batch(config['batch_size']*strategy.num_replicas_in_sync).prefetch(64)

        #learning rate scheduler 
        #lr_fn = partial(linear_scheduler, dim_embed = config['embed_layer_size'])
        

        model = transformer(length = config['length'],
            channels=config['channel'],
            num_heads=config['num_heads'],
            dropout_rate = config['dropout'],
            attention_dropout_rate = config['attention_dropout'],
            embed_dim =config['embed_layer_size'],
            mlp_dim = config['fc_layer_size'],
            num_layers = config['num_layers'], 
            output_bias = initial_bias)


        #processing train data 
        # model = tf.keras.saving.load_model()
        #lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = 0.002,decay_steps=10)
        model.compile(
            loss= BinaryCrossentropy(),
            optimizer=Adam(learning_rate=config['learning_rate']),
            metrics=[Recall(), Precision(),F1_Score()],
            )
        
        checkpoint_filepath = os.path.join(os.getcwd(), work_dir+"{val_loss:02f}_{val_f1_score:.2f}_{val_recall:.2f}.hdf5")
        model_checkpoint = ModelCheckpoint(filepath = checkpoint_filepath, 
                                            save_weights_only = True, 
                                            monitor = 'val_loss',
                                            mode = 'min', 
                                            save_best_only = True, 
                                            verbose = 1)

        #log_dir = "logs/"  # Specify the directory where TensorBoard logs will be saved
        model.summary()
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(
            x = X_train,
            y = y_train,
            epochs=config['epochs'],
            validation_data=(X_val,y_val),
            shuffle = True,
            callbacks=[
                # LearningRateScheduler(lr_fn(base_lr=config['learning_rate'], total_steps=config['epochs'], warmup_steps=config['warmup_steps']), verbose = 1),
                model_checkpoint, 
                EarlyStopping(monitor="val_loss", mode='min', min_delta=0.0001, patience=10, restore_best_weights = True),
                SaveLossCurve(work_dir),
                
            ],
            verbose=1,
            batch_size = config['batch_size'], 
            class_weight=class_weight
            )

        evaluation = model.evaluate(x=X_test,
            y=y_test,
            batch_size=config['batch_size'],
            verbose='auto')
        # y_prob = model.predict(X_test)
        # print(np.unique(y_prob))

        # print(evaluation)
        print("==========================")
        #print(evaluation)
        print("Test Loss:", evaluation[0])
        print("Test F1-Score:", evaluation[3])
        
        model.save(work_dir+"transformer_model.h5")
