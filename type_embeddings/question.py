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

        self.question_count = 16

        self.type_embeddings = TE()

        self.compose = layers.LSTM(type_embedding_dim, name='compose')

        self.subtype = KInputsNN(k=2, action='subtype')
        self.has_as_member = KInputsNN(k=2, action='member')
        self.has_as_parameter = KInputsNN(k=2, action='parameter')
        self.returns = KInputsNN(k=2, action='return')

        self.ok = [0 for _ in range(self.question_count)]
        self.cnt = [0 for _ in range(self.question_count)]

    def call(self, inputs, **kwargs):
        class_embeddings, function_embeddings, member_function_embeddings = self.type_embeddings.call(inputs)

        class_id_count = len(inputs["classes"])
        function_id_count = len(inputs["functions"])

        while True:
            question_id = random.choices([i for i in range(self.question_count)], )[0]

            # (A, B): A is subtype B
            if question_id == 0:
                derived_id = random.randrange(class_id_count)

                possible_super_types = inputs["classes"][derived_id]["superTypes"]
                if len(possible_super_types) == 0:
                    continue

                true_base_id = random.choice(possible_super_types)

                actual = self.subtype([class_embeddings[derived_id], class_embeddings[true_base_id]])
                real = 1.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, B): A is NOT subtype B
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
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, B): A is NOT subtype B, because A is function
            if question_id == 2:
                if class_id_count == 0 or function_id_count == 0:
                    continue

                derived_id = random.randrange(function_id_count)
                base_id = random.randrange(class_id_count)

                actual = self.subtype([function_embeddings[derived_id], class_embeddings[base_id]])
                real = 0.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, B): A is NOT subtype B, because B is function
            if question_id == 3:
                if class_id_count == 0 or function_id_count == 0:
                    continue

                derived_id = random.randrange(class_id_count)
                base_id = random.randrange(function_id_count)

                actual = self.subtype([class_embeddings[derived_id], function_embeddings[base_id]])
                real = 0.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1
                
                return actual, real

            # (A, B): A is NOT subtype B, because A and B are functions
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
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, X): A contains property with type X as member
            if question_id == 5:
                class_a = random.randrange(class_id_count)

                all_properties = self.get_all_properties(class_a, inputs)
                if len(all_properties) == 0:
                    continue

                property_id = random.choice(all_properties)

                actual = self.has_as_member([class_embeddings[class_a], class_embeddings[property_id]])
                real = 1.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, X): A contains property with type Y as member, Y is subtype of X
            if question_id == 6:
                class_a = random.randrange(class_id_count)

                all_properties = self.get_all_properties(class_a, inputs)
                if len(all_properties) == 0:
                    continue

                all_subtypes = []
                for p in all_properties:
                    all_subtypes.extend(self.subtypes_of(p, inputs))
                unused_subtypes = list(filter(lambda p: p not in all_properties, all_subtypes))

                if len(unused_subtypes) == 0:
                    continue

                property_id = random.choice(unused_subtypes)

                actual = self.has_as_member([class_embeddings[class_a], class_embeddings[property_id]])
                real = 1.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, X): A NOT contains property with type X as member
            if question_id == 7:
                class_a = random.randrange(class_id_count)

                all_properties = self.get_all_properties(class_a, inputs)
                all_subtypes = []
                for p in all_properties:
                    all_subtypes.extend(self.subtypes_of(p, inputs))
                unused_types = list(filter(lambda p: p not in all_properties and p not in all_subtypes, [i for i in range(class_id_count)]))

                if len(unused_types) == 0:
                    continue

                property_id = random.choice(unused_types)

                actual = self.has_as_member([class_embeddings[class_a], class_embeddings[property_id]])
                real = 0.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, B): A contains function with type B
            if question_id == 8:
                class_a = random.randrange(class_id_count)

                all_functions = member_function_embeddings[class_a]
                for supertype in inputs["classes"][class_a]["superTypes"]:
                    all_functions.extend(member_function_embeddings[supertype])

                if len(all_functions) == 0:
                    continue

                function_embedding = random.choice(all_functions)

                actual = self.has_as_member([class_embeddings[class_a], function_embedding])
                real = 1.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, B): A NOT contains function with type B
            if question_id == 9:
                class_a = random.randrange(class_id_count)

                current_functions = inputs["classes"][class_a]["functions"]
                for supertype in inputs["classes"][class_a]["superTypes"]:
                    current_functions.extend(inputs["classes"][supertype]["functions"])

                chosen_other_function_id = None
                for index, other_description in enumerate(inputs["functions"]):
                    chosen = False
                    for current_description in current_functions:
                        if self.functions_are_similar(other_description, current_description):
                            break
                    if chosen:
                        chosen_other_function_id = index
                        break

                if chosen_other_function_id is None:
                    continue

                function_embedding = inputs["functions"][chosen_other_function_id]

                actual = self.has_as_member([class_embeddings[class_a], function_embedding])
                real = 0.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, X): A contains parameter with type X
            if question_id == 10:
                function_a = random.randrange(function_id_count)

                all_parameters = inputs["functions"][function_a]["parameters"]
                if len(all_parameters) == 0:
                    continue

                parameter_id = random.choice(all_parameters)

                actual = self.has_as_parameter([function_embeddings[function_a], class_embeddings[parameter_id]])
                real = 1.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, X): A contains parameter with type Y, Y is subtype of X
            if question_id == 11:
                function_a = random.randrange(function_id_count)

                all_parameters = inputs["functions"][function_a]["parameters"]
                if len(all_parameters) == 0:
                    continue

                all_subtypes = []
                for p in all_parameters:
                    all_subtypes.extend(self.subtypes_of(p, inputs))
                unused_subtypes = list(filter(lambda p: p not in all_parameters, all_subtypes))

                if len(unused_subtypes) == 0:
                    continue

                parameter_id = random.choice(unused_subtypes)

                actual = self.has_as_parameter([function_embeddings[function_a], class_embeddings[parameter_id]])
                real = 1.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, X): A NOT contains parameter with type X
            if question_id == 12:
                function_a = random.randrange(function_id_count)

                all_parameters = inputs["functions"][function_a]["parameters"]
                all_subtypes = []
                for p in all_parameters:
                    all_subtypes.extend(self.subtypes_of(p, inputs))
                unused_types = list(filter(lambda p: p not in all_parameters and p not in all_subtypes, [i for i in range(class_id_count)]))

                if len(unused_types) == 0:
                    continue

                parameter_id = random.choice(unused_types)

                actual = self.has_as_parameter([function_embeddings[function_a], class_embeddings[parameter_id]])
                real = 0.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, B): A return B
            if question_id == 13:
                function_a = random.randrange(function_id_count)

                return_type = inputs["functions"][function_a]["returnType"]

                actual = self.returns([function_embeddings[function_a], class_embeddings[return_type]])
                real = 1.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, B): A return C, C is subtype of B
            if question_id == 14:
                function_a = random.randrange(function_id_count)

                return_type = inputs["functions"][function_a]["returnType"]
                all_subtypes = self.subtypes_of(return_type, inputs)
                if len(all_subtypes) == 0:
                    continue

                return_type = random.choice(all_subtypes)

                actual = self.returns([function_embeddings[function_a], class_embeddings[return_type]])
                real = 1.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

            # (A, B): A NOT return B
            if question_id == 15:
                function_a = random.randrange(function_id_count)

                return_type = inputs["functions"][function_a]["returnType"]
                all_subtypes = self.subtypes_of(return_type, inputs)
                unused_types = list(filter(lambda p: p != return_type and p not in all_subtypes, [i for i in range(class_id_count)]))

                if len(unused_types) == 0:
                    continue

                return_type = random.choice(unused_types)

                actual = self.returns([function_embeddings[function_a], class_embeddings[return_type]])
                real = 0.0

                self.cnt[question_id] += 1
                if tensorflow.abs(actual - real) < 0.5: 
                    self.ok[question_id] += 1

                return actual, real

    def subtypes_of(self, class_id, inputs):
        subtypes = []
        for klass_id, klass in enumerate(inputs["classes"]):
            if class_id in klass["superTypes"]:
                subtypes.append(klass_id)
        return subtypes

    def get_all_properties(self, class_id, inputs):
        result = []

        def walk_through_supertypes(current):
            result.extend(current["properties"])
            for supertype in current["superTypes"]:
                walk_through_supertypes(inputs["classes"][supertype])

        walk_through_supertypes(inputs["classes"][class_id])
        return result

    def randomly_change_to_subtype(self, property_ids, inputs):
        for i in range(len(property_ids)):
            if random.random() < 0.1:
                property_ids[i] = random.choice(self.subtypes_of(property_ids[i], inputs))
        return property_ids

    def functions_are_similar(self, a, b):
        if a["returnType"] != b["returnType"]:
            return False

        a_params = a["parameters"]
        b_params = b["parameters"]
        if len(a_params) != len(b_params):
            return False

        return sorted(a_params) == sorted(b_params)


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
