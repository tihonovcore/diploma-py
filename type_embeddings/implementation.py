import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from configuration import Configuration


class TE(keras.Model):
    def __init__(
            self,
            basic_types_count=Configuration.basic_types_count,
            type_embedding_dim=Configuration.type_embedding_dim,
            **kwargs
    ):
        super(TE, self).__init__(name='type_embeddings', **kwargs)

        self.basic = layers.Embedding(basic_types_count, type_embedding_dim)

        self.embed_function_parameters = layers.LSTM(type_embedding_dim)
        self.embed_members = layers.LSTM(type_embedding_dim)
        self.embed_super_types = layers.LSTM(type_embedding_dim)

        initializer = tf.random_normal_initializer()
        self.empty_parameters_list = tf.Variable(
            initial_value=initializer(shape=(type_embedding_dim, ), dtype="float32"),
            trainable=True,
        )
        self.empty_members_list = tf.Variable(
            initial_value=initializer(shape=(type_embedding_dim, ), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        class_embeddings = [None for _ in range(len(inputs["classes"]))]

        for klass in inputs["classes"]:
            self.walk(klass["id"], inputs, class_embeddings)

        return tf.convert_to_tensor(class_embeddings)

    def walk(self, current_id, inputs, class_embeddings):
        if class_embeddings[current_id] is not None:
            return

        description = inputs["classes"][current_id]

        if description["isBasic"]:
            class_embeddings[current_id] = self.get_embedding_for_basic(description["name"])
            return

        for dependency_id in description["dependencies"]:
            self.walk(dependency_id, inputs, class_embeddings)

        property_embeddings = [self.empty_members_list]
        for property_id in description["properties"]:
            property_embeddings.append(class_embeddings[property_id])
        property_embeddings = tf.convert_to_tensor(property_embeddings)
        property_embeddings = tf.reshape(property_embeddings, [1] + property_embeddings.shape)

        # todo: embed functions

        embedding_for_this_class = self.embed_members(property_embeddings)
        embedding_for_this_class = tf.reshape(embedding_for_this_class, [Configuration.type_embedding_dim])

        super_types_embeddings = []
        for super_type_id in description["superTypes"]:
            super_types_embeddings.append(class_embeddings[super_type_id])
        super_types_embeddings.append(embedding_for_this_class)

        super_types_embeddings = tf.convert_to_tensor(super_types_embeddings)
        super_types_embeddings = tf.reshape(super_types_embeddings, [1] + super_types_embeddings.shape)

        embedding_for_this_class = self.embed_super_types(super_types_embeddings)
        embedding_for_this_class = tf.reshape(embedding_for_this_class, [Configuration.type_embedding_dim])
        class_embeddings[current_id] = embedding_for_this_class

    def get_embedding_for_basic(self, name):
        index = ["kotlin.Any", "kotlin.Byte", "kotlin.Char", "kotlin.Double",
                 "kotlin.Float", "kotlin.Int", "kotlin.Long", "kotlin.Short",
                 "kotlin.CharSequence", "kotlin.Boolean", "kotlin.Unit"].index(name)

        return self.basic(tf.constant(index))
