from memory_profiler import profile
from tensorflow import keras
from tensorflow.keras import layers

from configuration import Configuration


class Encoder(layers.Layer):
    def __init__(
            self,
            path_embedding_dim=Configuration.path_embedding_dim,
            encoder_attention_heads_count=Configuration.encoder_attention_heads_count,
            encoder_ff_first_layer_dim=Configuration.encoder_ff_first_layer_dim,
            rate=0.1
    ):
        super(Encoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=encoder_attention_heads_count, key_dim=path_embedding_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(encoder_ff_first_layer_dim, activation="relu"), layers.Dense(path_embedding_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    # @profile
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, attention_mask=self.mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
