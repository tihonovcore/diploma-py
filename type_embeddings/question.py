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

        self.question_count = 5

        self.type_embeddings = TE()

        self.compose = layers.LSTM(type_embedding_dim)

        self.subtype = KInputsNN(k=2)
        self.has_as_members = KInputsNN(k=2)
        self.returns = KInputsNN(k=3)

        self.cnt = [0 for _ in range(self.question_count)]

    def call(self, inputs, **kwargs):
        class_embeddings, function_embeddings, member_function_embeddings = self.type_embeddings.call(inputs)

        class_id_count = len(inputs["classes"])
        function_id_count = len(inputs["functions"])

        while True:
            question_id = random.choices([i for i in range(self.question_count)], weights=[4, 1, 1, 1, 1])[0]

            # A is subtype B
            if question_id == 0:
                derived_id = random.randrange(class_id_count)

                possible_super_types = inputs["classes"][derived_id]["superTypes"]
                if len(possible_super_types) == 0:
                    continue

                true_base_id = random.choice(possible_super_types)

                actual = self.subtype([class_embeddings[derived_id], class_embeddings[true_base_id]])
                real = 1.0

                self.cnt[question_id] += 1
                return [actual], [real]

            # A is NOT subtype B
            if question_id == 1:
                derived_id = random.randrange(class_id_count)

                possible_super_types = inputs["classes"][derived_id]["superTypes"]
                impossible_super_types = list(filter(lambda t: t not in possible_super_types, [i for i in range(class_id_count)]))
                if len(impossible_super_types) == 0:
                    continue

                wrong_base_id = random.choice(impossible_super_types)

                actual = self.subtype([class_embeddings[derived_id], class_embeddings[wrong_base_id]])
                real = 0.0

                self.cnt[question_id] += 1
                return [actual], [real]

            # A is NOT subtype B, because A is function
            if question_id == 2:
                if class_id_count == 0 or function_id_count == 0:
                    continue

                derived_id = random.randrange(function_id_count)
                base_id = random.randrange(class_id_count)

                actual = self.subtype([function_embeddings[derived_id], class_embeddings[base_id]])
                real = 0.0

                self.cnt[question_id] += 1
                return [actual], [real]

            # A is NOT subtype B, because B is function
            if question_id == 3:
                if class_id_count == 0 or function_id_count == 0:
                    continue

                derived_id = random.randrange(class_id_count)
                base_id = random.randrange(function_id_count)

                actual = self.subtype([class_embeddings[derived_id], function_embeddings[base_id]])
                real = 0.0

                self.cnt[question_id] += 1
                return [actual], [real]

            # A is NOT subtype B, because A and B are functions
            if question_id == 4:
                if function_id_count <= 1:
                    continue

                derived_id = random.randrange(function_id_count)
                base_id = random.randrange(function_id_count)

                if base_id == derived_id:
                    base_id = (base_id + 1) % function_id_count

                actual = self.subtype([function_embeddings[derived_id], function_embeddings[base_id]])
                real = 0.0

                self.cnt[question_id] += 1
                return [actual], [real]

    def subtypes_of(self, x, inputs):
        pass

    def super_properties(self, x, inputs):
        pass


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
