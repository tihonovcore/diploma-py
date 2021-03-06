import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from configuration import Configuration


class Encoder(layers.Layer):
    def __init__(
            self,
            type_embedding_dim=Configuration.type_embedding_dim,
            encoder_attention_heads_count=4,
            encoder_ff_first_layer_dim=Configuration.encoder_ff_first_layer_dim,
            rate=0.1
    ):
        super(Encoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=encoder_attention_heads_count, key_dim=type_embedding_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(encoder_ff_first_layer_dim, activation="relu"), layers.Dense(type_embedding_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.pool = layers.GlobalAveragePooling1D()

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        result = self.layernorm2(out1 + ffn_output)
        return self.pool(result)
