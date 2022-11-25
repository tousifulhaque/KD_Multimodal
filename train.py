# from  wandb_config import sweep_config
# import wandb
import tensorflow as tf
from model import Transformer
from tensorflow.keras.optimizers import Adam
# from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from tensorflow.keras.losses import BinaryCrossentropy
from loss import smoothed_sparse_categorical_crossentropy
from sklearn.model_selection import train_test_split 
from lr_scheduler  import cosine_schedule
import numpy as np
# from wandb.keras import WandbCallback
import warnings
import logging
import os

config = {
      'epochs': 50,
      'num_layers':  3,
      'embed_layer_size': 128,
      'fc_layer_size': 256,
      'num_heads': 6,
      'dropout': 0.1,
      'attention_dropout': 0.1,
      'optimizer': 'adam',
      'amsgrad': False,
      'label_smoothing': 0.1,
      'learning_rate': 1e-3,
      #'weight_decay': {
      #    'values': [2.5e-4, 1e-4, 5e-5, 1e-5]
      'warmup_steps': 10,
      'batch_size': 64,
      'global_clipnorm': 3.0
}

class PrintLR(Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"lr": self.model.optimizer.lr.numpy()}, commit=False)

def train( input_shape = None):
    config = {
      'epochs': 50,
      'num_layers':  3,
      'embed_layer_size': 128,
      'global_clipnorm' : 3.0,
      'fc_layer_size': 256,
      'num_heads': 2,
      'dropout': 0.1,
      'attention_dropout': 0.1,
      'optimizer': 'adam',
      'amsgrad': False,
      'label_smoothing': 0.1,
      'learning_rate': 1e-3,
      #'weight_decay': {
      #    'values': [2.5e-4, 1e-4, 5e-5, 1e-5]
      'warmup_steps': 10,
      'batch_size': 8}
  
  # with wandb.init(config=config):
  #   tf.debugging.set_log_device_placement(True)
  #   config = wandb.config
    
    # Generate new model
    model = Transformer(
      num_layers=config['num_layers'],
      embed_dim=config['embed_layer_size']
      ,
      mlp_dim=config['fc_layer_size'],
      num_heads=config['num_heads'],
      num_classes=2,
      dropout_rate=config['dropout'],
      attention_dropout_rate=config['attention_dropout'],
    )

    # adapt on training dataset - must be before model.compile !!!
    model.input_norm.adapt(X_train, batch_size=config['batch_size'])
    # print(model.input_norm.variables)

    # Select optimizer
    if config['optimizer'] == "adam":
      optim = Adam(
          global_clipnorm=config['global_clipnorm'],
          amsgrad=config['amsgrad'],
      )
    # elif config.optimizer == "adamw":
    #   optim = AdamW(
    #       weight_decay=config.weight_decay,
    #       amsgrad=config.amsgrad,
    #       global_clipnorm=config.global_clipnorm,
    #       exclude_from_weight_decay=["position"]
    #   )
    else:
      raise ValueError("The used optimizer is not in list of available")

    model.compile(
      loss= BinaryCrossentropy(label_smoothing=config['label_smoothing']),
      optimizer=optim,
      metrics=["accuracy"],
    )


    # Train model
    model.fit(
      X_train,
      y_train,
      batch_size=config['batch_size'],
      epochs=config['epochs'],
      validation_data=(X_val, y_val),
      callbacks=[
        LearningRateScheduler(cosine_schedule(base_lr=config['learning_rate'], total_steps=config['epochs'], warmup_steps=config['warmup_steps'])),
        # PrintLR(),
        # WandbCallback(monitor="val_accuracy", mode='max', save_weights_only=True),
        EarlyStopping(monitor="val_accuracy", mode='max', min_delta=0.001, patience=5),
      ],
      verbose=1
    )

    model.summary()
        

if __name__ == "__main__":

    # wandb.login()
    # warnings.simplefilter('ignore')
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    # logging.getLogger('tensorflow').setLevel(logging.FATAL)

    # load dataset
    dataset_path = os.path.join(os.getcwd(), 'fall_detection_dataset.npz')
    f = np.load(dataset_path)
    signals = f['trials']

    labels = f['labels']

    # split to train-test
    X_train, X_test, y_train, y_test = train_test_split(
        signals, labels, test_size=0.15, random_state=9, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=9, stratify=y_train
    )

    # # sweep_id = wandb.sweep(sweep_config, project="KD_Transformer")
    # print(y_train.shape)
    with tf.device('/gpu:0'):

    #   # wandb.agent(sweep_id, train, count=32)
      train()

