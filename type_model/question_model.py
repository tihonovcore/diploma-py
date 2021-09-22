import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from configuration import Configuration
from type_model.generate_questions import QuestionSample
from type_model.implementation import TE


class QuestionModel(keras.Model):
    def __init__(
            self,
            type_embedding_dim=Configuration.type_embedding_dim,
            mode=Configuration.recurrent_mode,
            **kwargs
    ):
        super(QuestionModel, self).__init__(name='question_model', **kwargs)

        self.question_count = Configuration.question_type_count

        self.type_embeddings = TE(mode=mode)

        self.compose = layers.LSTM(type_embedding_dim, name='compose')

        self.subtype = KInputsNN(k=2, action='subtype')
        self.has_as_member = KInputsNN(k=2, action='member')
        self.has_as_parameter = KInputsNN(k=2, action='parameter')
        self.returns = KInputsNN(k=2, action='return')

        self.ok = [0 for _ in range(self.question_count)]
        self.cnt = [0 for _ in range(self.question_count)]

    def call(self, inputs, **kwargs):
        inputs, question = inputs
        class_embeddings, function_embeddings, member_function_embeddings = self.type_embeddings.call(inputs)

        question: QuestionSample
        question_id = question.question_id
        real = question.true_answer

        # (A, B): A is subtype B
        if question_id == 0:
            derived_id, true_base_id = question.params
            actual = self.subtype([class_embeddings[derived_id], class_embeddings[true_base_id]])

        # (A, B): A is NOT subtype B
        if question_id == 1:
            derived_id, wrong_base_id = question.params
            actual = self.subtype([class_embeddings[derived_id], class_embeddings[wrong_base_id]])

        # (A, B): A is NOT subtype B, because A is function
        if question_id == 2:
            derived_id, base_id = question.params
            actual = self.subtype([function_embeddings[derived_id], class_embeddings[base_id]])

        # (A, B): A is NOT subtype B, because B is function
        if question_id == 3:
            derived_id, base_id = question.params
            actual = self.subtype([class_embeddings[derived_id], function_embeddings[base_id]])

        # (A, B): A is NOT subtype B, because A and B are functions
        if question_id == 4:
            derived_id, base_id = question.params
            actual = self.subtype([function_embeddings[derived_id], function_embeddings[base_id]])

        # (A, X): A contains property with type X as member
        if question_id == 5:
            class_a, property_id = question.params
            actual = self.has_as_member([class_embeddings[class_a], class_embeddings[property_id]])

        # (A, X): A NOT contains property with type X as member
        if question_id == 6:
            class_a, property_id = question.params
            actual = self.has_as_member([class_embeddings[class_a], class_embeddings[property_id]])

        # (A, B): A contains function with type B
        if question_id == 7:
            class_a, container_id, function_index = question.params
            actual = self.has_as_member([class_embeddings[class_a], member_function_embeddings[container_id][function_index]])

        # (A, B): A NOT contains function with type B
        if question_id == 8:
            class_a, chosen_other_function_id = question.params
            actual = self.has_as_member([class_embeddings[class_a], function_embeddings[chosen_other_function_id]])

        # (A, X): A contains parameter with type X
        if question_id == 9:
            function_a, parameter_id = question.params
            actual = self.has_as_parameter([function_embeddings[function_a], class_embeddings[parameter_id]])

        # (A, X): A contains parameter with type Y, Y is supertype of X
        if question_id == 10:
            chosen_function_id, chosen_parameter_id = question.params
            actual = self.has_as_parameter([function_embeddings[chosen_function_id], class_embeddings[chosen_parameter_id]])

        # (A, X): A NOT contains parameter with type X
        if question_id == 11:
            function_a, parameter_id = question.params
            actual = self.has_as_parameter([function_embeddings[function_a], class_embeddings[parameter_id]])

        # (A, B): A return B
        if question_id == 12:
            function_a, return_type = question.params
            actual = self.returns([function_embeddings[function_a], class_embeddings[return_type]])

        # (A, B): A return C, C is subtype of B
        if question_id == 13:
            function_a, return_type = question.params
            actual = self.returns([function_embeddings[function_a], class_embeddings[return_type]])

        # (A, B): A NOT return B
        if question_id == 14:
            function_a, return_type = question.params
            actual = self.returns([function_embeddings[function_a], class_embeddings[return_type]])

        self.cnt[question_id] += 1
        if tensorflow.abs(actual - real) < 0.5:
            self.ok[question_id] += 1

        return actual, real


class KInputsNN(keras.Model):
    def __init__(
            self,
            action,
            k=1,
            type_embedding_dim=Configuration.type_embedding_dim,
            **kwargs
    ):
        super(KInputsNN, self).__init__(name='two_inputs_neural_network action:' + action, **kwargs)

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
