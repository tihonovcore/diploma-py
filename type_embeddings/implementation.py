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
        self.function_from_params_and_return = tf.Variable(
            initial_value=initializer(shape=(2 * type_embedding_dim, type_embedding_dim), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        class_embeddings = [None for _ in range(len(inputs["classes"]))]

        for klass in inputs["classes"]:
            self.walk(klass["id"], inputs, class_embeddings)

        function_embeddings = []
        for function_description in inputs["functions"]:
            function_embeddings.append(self.get_embedding_for_function(function_description, class_embeddings))

        return tf.convert_to_tensor(class_embeddings), tf.convert_to_tensor(function_embeddings)

    def walk(self, current_id, inputs, class_embeddings):
        if class_embeddings[current_id] is not None:
            return

        description = inputs["classes"][current_id]

        if description["isBasic"]:
            class_embeddings[current_id] = self.get_embedding_for_basic(description["name"])
            return

        for dependency_id in description["dependencies"]:
            self.walk(dependency_id, inputs, class_embeddings)

        member_embeddings = [self.empty_members_list]
        for property_id in description["properties"]:
            member_embeddings.append(class_embeddings[property_id])
        for function_description in description["functions"]:
            member_embeddings.append(self.get_embedding_for_function(function_description, class_embeddings))

        member_embeddings = tf.convert_to_tensor(member_embeddings)
        member_embeddings = tf.reshape(member_embeddings, [1] + member_embeddings.shape)

        embedding_for_this_class = self.embed_members(member_embeddings)
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

    def get_embedding_for_function(self, description, class_embeddings):
        parameter_embeddings = [self.empty_parameters_list]
        for parameter_id in description["parameters"]:
            parameter_embeddings.append(class_embeddings[parameter_id])

        parameter_embeddings = tf.convert_to_tensor(parameter_embeddings)
        parameter_embeddings = tf.reshape(parameter_embeddings, [1] + parameter_embeddings.shape)

        embedding_for_parameter_list = self.embed_function_parameters(parameter_embeddings)
        embedding_for_parameter_list = tf.reshape(embedding_for_parameter_list, [Configuration.type_embedding_dim])

        return_type_id = description["returnType"]
        embedding_for_return_type = class_embeddings[return_type_id]

        result = tf.concat([embedding_for_parameter_list, embedding_for_return_type], axis=0)
        result = tf.reshape(result, [1, 2 * Configuration.type_embedding_dim])
        result = tf.matmul(result, self.function_from_params_and_return)
        result = tf.reshape(result, [Configuration.type_embedding_dim])
        return result
