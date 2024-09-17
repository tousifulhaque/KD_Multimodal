import os

#tensorflow imports 
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, LayerNormalization,GlobalAveragePooling1D, BatchNormalization 
from tensorflow.keras.layers import Conv1D, Dropout, MultiHeadAttention, Layer, Embedding, DepthwiseConv1D
from tensorflow.keras.initializers import TruncatedNormal

def conv_encoder(x, length,embed_dim, mlp_dim, num_heads, attn_drop_rate, num_channels, drop_rate):
    

    #attention_layer
    residual = x
    x = LayerNormalization(epsilon = 1e-6)(x)
    x = Dense(units = mlp_dim, activation= 'relu')(x)
    x = Dropout(rate =drop_rate)(x)
    x = Dense(units=embed_dim, activation='relu')(x)
    x = Dropout(drop_rate)(x)
    x = Add()([residual, (0.5*x)])
    x = LayerNormalization(epsilon = 1e-6)(x)


    #y = BatchNormalization()(x)
    residual = x
    x = MultiHeadAttention(num_heads = num_heads,key_dim =embed_dim ,dropout = attn_drop_rate, kernel_initializer = TruncatedNormal(stddev = 0.02))(query = x,value = x,key = x,training = True)
    x = Dropout(drop_rate)(x)
    x = Add()([residual, x])
    x= LayerNormalization(epsilon = 1e-6)(x)
    #es = BatchNormalization()(res)
    residual = x 

    x = Conv1D(filters=embed_dim//2, kernel_size=1, padding = 'same', strides=1 ,activation = 'relu')(x)
    x = BatchNormalization()(x)


    x = DepthwiseConv1D(kernel_size=length - 2, strides=1,padding = 'same', activation = tf.keras.activations.swish)(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters = embed_dim, kernel_size=1, padding = 'same', strides=1 ,activation = 'relu')(x)
    x = Dropout(drop_rate)(x)
    x = Add()([residual, x])

    # Pointwise Convolution

#     #mlp_layer
    residual = x
    x = LayerNormalization(epsilon = 1e-6)(x)
    x = Dense(units = mlp_dim,  activation = 'relu')(x)
    x = Dropout(rate = drop_rate)(x)
    x = Dense(units = embed_dim, activation = 'relu')(x)
    x = Dropout(rate = drop_rate)(x)
    x = Add()([residual,(0.5*x)])
    y = LayerNormalization(epsilon = 1e-6)(x)
    
    return x

