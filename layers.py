from tensorflow.keras.layers import Add, Dense, Dropout, MultiHeadAttention, LayerNormalization, Layer
from tensorflow.keras.initializers import TruncatedNormal
class PositionalEmbedding(Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.units = units

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
        x = x + self.position
        return self.dropout(x, training=training)


class Encoder(Layer):
    def __init__(
        self, embed_dim, mlp_dim, num_heads, dropout_rate,
        attention_dropout_rate, **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)
        
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

        self.dropout_0 = Dropout(rate = dropout_rate)
        self.dropout_1 = Dropout(rate = dropout_rate)

        self.norm_0 = LayerNormalization(epsilon = 1e-5)
        self.norm_1 = LayerNormalization(epsilon = 1e-5)

        self.add_0 = Add()
        self.add_1 = Add()
    
    def call(self, inputs, training):

        x = self.norm_0(inputs)
        x = self.mha(
            query = x, 
            value = x, 
            key = x,
            training = training
        )

        x = self.dropout_0(x, training= training)
        x = self.add_0([x, inputs])

        #MLP block 
        y = self.norm_1(x)
        y = self.dense_0(y)
        y = self.dense_1(y)
        y = self.dropout_1(y, training)

        return self.add_1([x, y])