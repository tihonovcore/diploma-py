import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from configuration import Configuration


class EncoderTransformer(layers.Layer):
    def __init__(self,):
        super(EncoderTransformer, self).__init__()

        self.body = keras.Sequential([Encoder(), Encoder()]) #, Encoder(), Encoder()])

    def call(self, inputs, **kwargs):
        result = self.body(inputs)
        result = result[0][0]
        result = tf.reshape(result, [1] + result.shape)
        return result


class Encoder(layers.Layer):
    def __init__(
            self,
            type_embedding_dim=Configuration.type_embedding_dim,
            encoder_attention_heads_count=1,
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

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
