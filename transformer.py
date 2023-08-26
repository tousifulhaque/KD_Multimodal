#global import 
import os

#tensorflow imports 
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, LayerNormalization,GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, Dropout, MultiHeadAttention, Layer, Embedding
from tensorflow.keras.initializers import TruncatedNormal

#local package import 
import numpy as np
import matplotlib.pyplot as plt


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

def encoder(x, embed_dim,attn_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate, length, channel):
    
    #attention_layer
    y = LayerNormalization(epsilon = 1e-6)(x)
    y = MultiHeadAttention(num_heads = num_heads,key_dim = attn_dim,dropout = attention_dropout_rate, kernel_initializer = TruncatedNormal(stddev = 0.02))(query = x,value = x,key = x,training = True)
    y = Dropout(rate = dropout_rate)(y)
    y = x + y
    y = LayerNormalization(epsilon = 1e-6)(y)
    res = y
    
    
    #mlp_layer
    
    y = Dense(units = mlp_dim, kernel_initializer = TruncatedNormal(stddev = 0.02))(y)
    y = Dropout(rate = dropout_rate)(y)
    y = Dense(units = x.shape[-1],kernel_initializer = TruncatedNormal(stddev = 0.02))(y)
    y = Dropout(rate = dropout_rate)(y)
    y = res + y
    y = LayerNormalization(epsilon = 1e-6)(y)
    
    return y
    
    

def transformer(length, channels,num_layers, embed_dim, attn_dim,mlp_dim, num_heads, dropout_rate, attention_dropout_rate):
    
    #initial normalization
    #pos_embed = get_positional_embedding(length, embed_dim)
    inputs= keras.Input(shape = (length, channels))
    x = inputs
    #x = Dense(embed_dim)(inputs)
    #x = Normalization()(inputs)
    #x = x + pos_embed
    #stacking encoder layers
    for _ in range(num_layers):
        x = encoder(x,embed_dim,attn_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate, length,channels)
    #x = LayerNormalization(epsilon=1e-5)(x)
    
    for dim in [8, 16]:
        x = Dense(dim, activation = 'relu')(x)
        x = Dropout(dropout_rate)(x)
        
    #pooling
    x = GlobalAveragePooling1D(data_format = 'channels_first')(x)
    
    #output
    output = Dense(1, kernel_initializer="zeros", activation = 'sigmoid')(x)
    
    return keras.Model(inputs, output)