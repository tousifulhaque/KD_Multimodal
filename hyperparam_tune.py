import optuna
import os
import tensorflow as tf
#from matplotlib import pyplot
import numpy as np
from numpy import argmax

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall, Precision


#local imports
from cfg import config, TRAIN, VALID, DATASET, WINDOW, STRIDE
from transformer import transformer
from utils import process_data , F1_Score, cosine_schedule
#config


#creating model
def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    if optimizer_name == "Adam":
        adam_lr = trial.suggest_float("adam_lr", 1e-5, 1e-1, log=True)
        return Adam(learning_rate=adam_lr)
    else:
        sgd_lr = trial.suggest_float("sgd_lr", 1e-5, 1e-1, log=True)
        sgd_momentum = trial.suggest_float("sgd_momentum", 1e-5, 1e-1, log=True)
        return SGD(learning_rate=sgd_lr, momentum=sgd_momentum)
    
def objective(trial):

    #hyperparameters
    #length = trial.suggest_categorical("length", [32, 64, 128, 256])
    num_layers = trial.suggest_categorical("num_layers", [1, 2 ,4, 8])
    dropout_rate = trial.suggest_float("dropout", .1, .4)
    attention_dropout = trial.suggest_float("attn_drop", .1, .4)
    embed_dim = trial.suggest_categorical("embed", [8, 16, 32, 64])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    fc_layer_size = trial.suggest_categorical("mlp_dim" , [16, 32, 64])
    batch_size = trial.suggest_categorical("batch_size", [16,32, 64,128])
    optimizer = create_optimizer(trial=trial)
    window = trial.suggest_categorical("window", [128, 256, 64])
    stride = trial.suggest_categorical("stride", [8, 16 , 32, 64])




    #processing train data
    data, labels = process_data(TRAIN, window, stride)
    X_train, y_train = data, labels.astype(np.int64)
    neg, pos = np.bincount(y_train)
    total = y_train.shape[0]
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    initial_bias = np.log([pos/neg])
    # train_file = np.load(TRAIN)
    # X_train , y_train = train_file['data'], train_file['labels']

    #processing val data 
    X_val , y_val = process_data(VALID, window, stride)
    X_val, y_val = X_val, y_val.astype(np.int64)
    #val_file = np.load(VALID)
    # X_val, y_val = val_file['data'], val_file['labels']
    model = transformer(length=window,
        channels=config['channel'],
        num_heads=num_heads,
        dropout_rate = dropout_rate,
        attention_dropout_rate = attention_dropout,
        embed_dim = embed_dim,
        mlp_dim = fc_layer_size,
        num_layers = num_layers, 
        output_bias=initial_bias)
    model.compile(
    loss= BinaryCrossentropy(),
    optimizer=optimizer,
    metrics=[F1_Score(), Recall(), Precision()],
    )

    # saved_dir = f'tmp/{DATASET}_{WINDOW}/'
    # if not os.path.exists(saved_dir):
    # os.mkdir(saved_dir)

    # checkpoint_filepath = os.path.join(os.getcwd(), saved_dir+"{epoch:02d}_{val_f1_score:.2f}.hdf5")
    # model_checkpoint = ModelCheckpoint(filepath = checkpoint_filepath, 
    #                                 save_weights_only = True, 
    #                                 monitor = 'val_loss',
    #                                 mode = 'min', 
    #                                 save_best_only = True, 
    #                                 verbose = True)
    #log_dir = "logs/"  # Specify the directory where TensorBoard logs will be saved
    model.summary()
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=50,
    validation_data=(X_val, y_val),
    shuffle = True,
    callbacks=[
        #LearningRateScheduler(cosine_schedule(base_lr=config['learning_rate'], total_steps=config['epochs'], warmup_steps=config['warmup_steps'])),
        EarlyStopping(monitor="val_loss", mode='min', min_delta=0.001, patience=5, restore_best_weights=True),
        #model_checkpoint
    ],
    verbose=1, 
    class_weight = class_weight
    )

    score = model.evaluate(X_val, y_val,verbose = 0)
    return score[0]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials = 200, timeout = 120)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
