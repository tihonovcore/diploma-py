import random

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from configuration import Configuration
from type_embeddings.implementation import TE


class Question(keras.Model):
    def __init__(
            self,
            type_embedding_dim=Configuration.type_embedding_dim,
            **kwargs
    ):
        super(Question, self).__init__(name='question_model', **kwargs)

        self.question_count = 2

        self.type_embeddings = TE()

        self.compose = layers.LSTM(type_embedding_dim)

        self.subtype = KInputsNN(k=2)
        self.has_as_members = KInputsNN(k=2)
        self.callable_with = KInputsNN(k=2)
        self.has_members = KInputsNN(k=3)

    def call(self, inputs, **kwargs):
        class_embeddings, function_embeddings, member_function_embeddings = self.type_embeddings.call(inputs)

        class_id_count = len(inputs["classes"])

        all_actual = []
        all_real = []

        question_id = random.randrange(self.question_count)

        # A is subtype B
        while question_id == 0:
            derived_id = random.randrange(class_id_count)

            possible_super_types = inputs["classes"][derived_id]["superTypes"]
            if len(possible_super_types) == 0:
                continue

            true_base_id = random.choice(possible_super_types)

            actual = self.subtype([class_embeddings[derived_id], class_embeddings[true_base_id]])
            real = 1.0

            all_actual.append(actual)
            all_real.append(real)
            break

        # A is NOT subtype B
        while question_id == 1:
            derived_id = random.randrange(class_id_count)

            possible_super_types = inputs["classes"][derived_id]["superTypes"]
            impossible_super_types = list(filter(lambda t: t not in possible_super_types, [i for i in range(class_id_count)]))
            if len(impossible_super_types) == 0:
                continue

            wrong_base_id = random.choice(impossible_super_types)

            actual = self.subtype([class_embeddings[derived_id], class_embeddings[wrong_base_id]])
            real = 0.0

            all_actual.append(actual)
            all_real.append(real)
            break

        # todo: A contains as members set X
        # todo: A NOT contains as members set X
        # todo: A is callable with set X
        # todo: A is NOT callable with set X
        # todo: A(X) is B
        # todo: A(X) is NOT B

        return all_actual, all_real


class KInputsNN(keras.Model):
    def __init__(
            self,
            k=1,
            type_embedding_dim=Configuration.type_embedding_dim,
            **kwargs
    ):
        super(KInputsNN, self).__init__(name='two_inputs_neural_network', **kwargs)

        self.k = k

        self.personal = [layers.Dense(type_embedding_dim) for _ in range(k)]

        self.fst_common_dense = layers.Dense(k * type_embedding_dim)
        self.snd_common_dense = layers.Dense(type_embedding_dim)

        self.question = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, **kwargs):
        inputs = list(map(lambda i: tensorflow.reshape(i, [1] + i.shape), inputs))
        inputs = list(map(lambda e: e[0](e[1]), zip(self.personal, inputs)))
        inputs = tensorflow.concat(inputs, axis=1)
        inputs = self.fst_common_dense(inputs)
        inputs = self.snd_common_dense(inputs)
        inputs = self.question(inputs)
        return inputs
