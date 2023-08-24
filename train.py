import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from matplotlib import pyplot
from numpy import argmax

train_X, train_y, test_Y, valid_X, valid_Y = fn.load_data(cfg.TRAIN, cfg.TEST, index=cfg.INDEX, header=cfg.HEADER)