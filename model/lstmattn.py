import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Attention
from tensorflow.keras.models import Model

def lstm_attn(n_timesteps, n_features, n_outputs):
    inputs = Input(shape=(n_timesteps, n_features))

    lstm_out = LSTM(32, return_sequences=True)(inputs)

    # Self-attention layer
    attention = Attention()([lstm_out, lstm_out])  # Self attention, Q = K = V
    attention = tf.keras.layers.GlobalAveragePooling1D()(attention)

    # Fully connected layers
    x = Dense(32, activation="relu")(attention)
    x = BatchNormalization()(x)
    outputs = Dense(n_outputs, activation="sigmoid")(x)
    return Model(inputs, outputs)
