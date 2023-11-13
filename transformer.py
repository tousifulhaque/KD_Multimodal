#global import 
import os

#tensorflow imports 
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, LayerNormalization,GlobalAveragePooling1D, BatchNormalization 
from tensorflow.keras.layers import Conv1D, Dropout, MultiHeadAttention, Layer, Embedding
from tensorflow.keras.initializers import TruncatedNormal

#local package import 
import numpy as np
#import matplotlib.pyplot as plt


# function to get sinusoidal positional embedding
def get_positional_embedding(seq_len,d_model, n = 10000):
    P = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in np.arange(int(d_model/2)):
            denominator = np.power(n, 2*i/d_model)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i + 1] = np.cos(k/denominator)
    P = P[np.newaxis, : ,:]
    return tf.Variable(P,trainable = False ,dtype = tf.float32)

def encoder(x, embed_dim, mlp_dim, num_heads, attn_drop_rate, num_channels, drop_rate):
    
    #attention_layer
    y = LayerNormalization(epsilon = 1e-6)(x)
    y = MultiHeadAttention(num_heads = num_heads,key_dim =embed_dim ,dropout = attn_drop_rate, kernel_initializer = TruncatedNormal(stddev = 0.02))(query = x,value = x,key = x,training = True)
    res = Add()([x ,y])
    y= LayerNormalization(epsilon = 1e-6)(res)
    
#     #mlp_layer
    
    y = Dense(units = mlp_dim, kernel_initializer = TruncatedNormal(stddev = 0.02))(res)
    y = Dropout(rate = drop_rate)(y)
    y = Dense(units = embed_dim, kernel_initializer = TruncatedNormal(stddev = 0.02))(y)
    y = Dropout(rate = drop_rate)(y)
    y = Add()([res,y])
    
    return y
    
    

def transformer(length, channels,num_layers,embed_dim,mlp_dim,num_heads, dropout_rate, attention_dropout_rate):
    
    #initial normalization
    #pos_embed = get_positional_embedding(length, embed_dim)
    inputs= keras.Input(shape = (length, channels))
    x = Conv1D(embed_dim//2, 3, activation='relu', data_format = 'channels_first', padding = "same")(inputs)
    x = Dense(embed_dim,  kernel_initializer = TruncatedNormal(stddev = 0.02))(x)
    
    #x = Normalization()(inputs)
    #x = Add()([x, pos_embed])
    #stacking encoder layers
    for _ in range(num_layers):
        x = encoder(x = x,embed_dim = embed_dim, mlp_dim = mlp_dim, num_heads = num_heads, attn_drop_rate = attention_dropout_rate, drop_rate = dropout_rate, num_channels = channels)
    #x = LayerNormalization(epsilon=1e-5)(x)
    
    x = GlobalAveragePooling1D(data_format = 'channels_last')(x)

    x = Dense(32, activation = 'relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation = 'relu')(x)    
#     #pooling
    
    
    #output
    output = Dense(1, kernel_initializer="zeros", activation = 'sigmoid')(x)
    
    return keras.Model(inputs, output)
