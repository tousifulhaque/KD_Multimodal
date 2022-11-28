from tensorflow.keras.layers import Dense , LayerNormalization, Normalization , Masking, GlobalAveragePooling1D
from layers import PositionalEmbedding, Encoder
from tensorflow.keras import Model
import tensorflow as tf


class Transformer(Model):
    def __init__(
        self,
        num_layers,
        embed_dim,
        mlp_dim,
        num_heads,
        num_classes,
        dropout_rate,
        attention_dropout_rate,
        **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)

        # Input (normalization of RAW measurements)
        self.input_norm = Normalization()
        # #Padding 
        # self.intf.keras.utils.pad_sequences(trials, maxlen= 512, value = float('-inf') , dtype = float, padding = 'post' )
        #Making Layer
        self.masking_layer = Masking(mask_value = 0.0)

        # Input
        self.pos_embs = PositionalEmbedding(embed_dim, dropout_rate)

        # Encoder
        self.e_layers = [
            Encoder(embed_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate)
            for _ in range(num_layers)
        ]

        # Output
        self.norm = LayerNormalization(epsilon=1e-5)
        self.pool = GlobalAveragePooling1D(data_format = 'channels_first')
        self.dense_0 = Dense(mlp_dim, activation = 'relu')
        self.final_layer = Dense(1, kernel_initializer="zeros", activation = 'sigmoid')

    def call(self, inputs, training = True):
        expanded_input = tf.cast(tf.tile(tf.expand_dims(inputs, axis=-2), [1, 1, 500,1]), tf.float32)
        self.masking_layer.build(expanded_input.shape)
        mask = self.masking_layer.compute_mask(expanded_input)
        x = self.input_norm(inputs) 
        # mask = self.masking_layer.compute_mask(inputs) 
        x = self.pos_embs(x, training=training)
        for layer in self.e_layers:
            x = layer(x, training=training , mask = mask)
        x = self.norm(x)
        x = self.pool(x)
        x = self.dense_0(x)

        x = self.final_layer(x)
        return x