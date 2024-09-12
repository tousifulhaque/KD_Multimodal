from tensorflow.keras.layers import Add, Dense, Conv1D, Dropout, MultiHeadAttention, LayerNormalization, Layer
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow as tf

class PositionalEmbedding(Layer):
    def __init__(self, units,dropout_rate,  **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.units = units
        self.conv_1 = Conv1D(filters  = units, kernel_size = 1)
        self.projection = Dense(units, kernel_initializer=TruncatedNormal(stddev=0.02))

        self.dropout = Dropout(rate=dropout_rate)

    def build(self, input_shape):
        super(PositionalEmbedding, self).build(input_shape)

        self.position = self.add_weight(
            name="position",
            shape=(1, input_shape[1], self.units),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs, training):
        x = self.projection(inputs)
        # x = self.conv_1(inputs)
        x = x + self.position
        return self.dropout(x, training=training)



class Encoder(Layer):
    def __init__(
        self, embed_dim, mlp_dim, num_heads, dropout_rate,
        attention_dropout_rate, **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)
        #embed_dim = 128, 
        #mlp_dim = 256
        self.mha = MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed_dim,
            dropout = attention_dropout_rate, 
            kernel_initializer = TruncatedNormal(stddev = 0.02)
        )

        self. dense_0 = Dense(
            units = mlp_dim, 
            activation = "gelu", 
            kernel_initializer = TruncatedNormal(stddev = 0.02)
        )

        self.dense_1 = Dense(
            units = embed_dim, 
            kernel_initializer = TruncatedNormal(stddev = 0.02)
        )

        self.conv_0 = Conv1D(filters = 4 , kernel_size = 1, activation = 'relu')
        self.conv_1 = Conv1D(filters  = embed_dim, kernel_size = 1)

        self.dropout_0 = Dropout(rate = dropout_rate)
        self.dropout_1 = Dropout(rate = dropout_rate)

        self.norm_0 = LayerNormalization(epsilon = 1e-6)
        self.norm_1 = LayerNormalization(epsilon = 1e-6)

        self.add_0 = Add()
        self.add_1 = Add()
    
    def call(self, inputs, training , mask):


        x = self.norm_0(inputs)
        x = self.mha(
            query = x, 
            value = x, 
            key = x,
            attention_mask = mask,
            training = training
        )

        x = self.dropout_0(x, training= training)
        x = self.add_0([x, inputs])

        #MLP block 
        y = self.norm_1(x)
        y = self.conv_0(y)
        y = self.dropout_1(y, training)
        y = self.conv_1(y)
        

        return self.add_1([x, y])



if __name__ == "__main__":
    postional_embedding = tf.keras_nlp.layers.PositionalEmbedding(sequence_length = 128)