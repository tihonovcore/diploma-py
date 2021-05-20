import random

from configuration import Configuration
from type_embeddings.utils import get_all_properties, subtypes_of, functions_are_similar


class QuestionSample:
    def __init__(self, question_id, params, true_answer, text):
        self.question_id = question_id
        self.params = params
        self.true_answer = true_answer
        self.text = text


def generate_questions(inputs, n: int):
    questions = []
    for _ in range(n):
        questions.append(generate_single_question(inputs))
    return questions


def generate_single_question(inputs):
    class_id_count = len(inputs["classes"])
    function_id_count = len(inputs["functions"])

    while True:
        question_id = random.choices(
            [i for i in range(Configuration.question_type_count)],
            weights=[
                1 / 8, 1 / 32, 1 / 32, 1 / 32, 1 / 32,
                1 / 24, 1 / 24, 1 / 16, 1 / 24, 1 / 16,
                1 / 16, 1 / 16, 1 / 8,
                1 / 16, 1 / 16, 1 / 8
            ],
            k=1
        )[0]

        # (A, B): A is subtype B
        if question_id == 0:
            derived_id = random.randrange(class_id_count)

            possible_super_types = inputs["classes"][derived_id]["superTypes"]
            if len(possible_super_types) == 0:
                continue

            true_base_id = random.choice(possible_super_types)

            A = inputs["classes"][derived_id]["name"]
            B = inputs["classes"][true_base_id]["name"]
            text = '%s is subtype %s - YES' % (A, B)
            return QuestionSample(question_id, [derived_id, true_base_id], 1.0, text)

        # (A, B): A is NOT subtype B
        if question_id == 1:
            derived_id = random.randrange(class_id_count)

            possible_super_types = inputs["classes"][derived_id]["superTypes"]
            impossible_super_types = list(
                filter(lambda t: t not in possible_super_types, [i for i in range(class_id_count)]))
            if len(impossible_super_types) == 0:
                continue

            wrong_base_id = random.choice(impossible_super_types)

            A = inputs["classes"][derived_id]["name"]
            B = inputs["classes"][wrong_base_id]["name"]
            text = '%s is subtype %s - NO' % (A, B)
            return QuestionSample(question_id, [derived_id, wrong_base_id], 0.0, text)

        # (A, B): A is NOT subtype B, because A is function
        if question_id == 2:
            if class_id_count == 0 or function_id_count == 0:
                continue

            derived_id = random.randrange(function_id_count)
            base_id = random.randrange(class_id_count)

            A = function(inputs["classes"], inputs["functions"][derived_id])
            B = inputs["classes"][base_id]["name"]
            text = '%s is subtype %s - NO' % (A, B)
            return QuestionSample(question_id, [derived_id, base_id], 0.0, text)

        # (A, B): A is NOT subtype B, because B is function
        if question_id == 3:
            if class_id_count == 0 or function_id_count == 0:
                continue

            derived_id = random.randrange(class_id_count)
            base_id = random.randrange(function_id_count)

            A = inputs["classes"][derived_id]["name"]
            B = function(inputs["classes"], inputs["functions"][base_id])
            text = '%s is subtype %s - NO' % (A, B)
            return QuestionSample(question_id, [derived_id, base_id], 0.0, text)

        # (A, B): A is NOT subtype B, because A and B are functions
        if question_id == 4:
            if function_id_count <= 1:
                continue

            derived_id = random.randrange(function_id_count)
            base_id = random.randrange(function_id_count)

            if base_id == derived_id:
                base_id = (base_id + 1) % function_id_count

            A = function(inputs["classes"], inputs["functions"][derived_id])
            B = function(inputs["classes"], inputs["functions"][base_id])
            text = '%s is subtype %s - NO' % (A, B)
            return QuestionSample(question_id, [derived_id, base_id], 0.0, text)

        # (A, X): A contains property with type X as member
        if question_id == 5:
            classes_with_properties = []
            for class_id in range(class_id_count):
                if len(get_all_properties(class_id, inputs)) != 0:
                    classes_with_properties.append(class_id)

            if len(classes_with_properties) == 0:
                continue

            class_a = random.choice(classes_with_properties)
            all_properties = get_all_properties(class_a, inputs)
            property_id = random.choice(all_properties)

            A = inputs["classes"][class_a]["name"]
            X = inputs["classes"][property_id]["name"]
            text = '%s contains member %s - YES' % (A, X)
            return QuestionSample(question_id, [class_a, property_id], 1.0, text)

        # todo: this is question for `write`. need question for `read`?
        # (A, X): A contains property with type Y as member, Y is supertype of X
        if question_id == 6:
            classes_with_properties = []
            for class_id in range(class_id_count):
                if len(get_all_properties(class_id, inputs)) != 0:
                    classes_with_properties.append(class_id)

            if len(classes_with_properties) == 0:
                continue

            random.shuffle(classes_with_properties)

            chosen_class_id = None
            chosen_property_id = None
            for class_id in classes_with_properties:
                all_properties = get_all_properties(class_id, inputs)

                all_subtypes = []
                for p in all_properties:
                    all_subtypes.extend(subtypes_of(p, inputs))
                unused_subtypes = list(filter(lambda p: p not in all_properties, all_subtypes))

                if len(unused_subtypes) != 0:
                    chosen_class_id = class_id
                    chosen_property_id = random.choice(unused_subtypes)

            if chosen_class_id is None:
                continue

            A = inputs["classes"][chosen_class_id]["name"]
            X = inputs["classes"][chosen_property_id]["name"]
            text = '%s contains member %s - YES' % (A, X)
            return QuestionSample(question_id, [chosen_class_id, chosen_property_id], 1.0, text)

        # (A, X): A NOT contains property with type X as member
        if question_id == 7:
            class_a = random.randrange(class_id_count)

            all_properties = get_all_properties(class_a, inputs)
            all_subtypes = []
            for p in all_properties:
                all_subtypes.extend(subtypes_of(p, inputs))
            unused_types = list(filter(lambda p: p not in all_properties and p not in all_subtypes, [i for i in range(class_id_count)]))

            if len(unused_types) == 0:
                continue

            property_id = random.choice(unused_types)

            A = inputs["classes"][class_a]["name"]
            X = inputs["classes"][property_id]["name"]
            text = '%s contains member %s - NO' % (A, X)
            return QuestionSample(question_id, [class_a, property_id], 0.0, text)

        # (A, B): A contains function with type B
        if question_id == 8:
            class_a = random.randrange(class_id_count)

            member_function_containers = [class_a] + inputs["classes"][class_a]["superTypes"]
            not_empty_containers = list(filter(lambda c: len(inputs["classes"][c]["functions"]) != 0, member_function_containers))

            if len(not_empty_containers) == 0:
                continue

            container_id = random.choice(not_empty_containers)
            function_index = random.randrange(len(inputs["classes"][container_id]["functions"]))

            A = inputs["classes"][class_a]["name"]
            B = function(inputs["classes"], inputs["classes"][container_id]["functions"][function_index])
            text = '%s contains member %s - YES' % (A, B)
            return QuestionSample(question_id, [class_a, container_id, function_index], 1.0, text)

        # (A, B): A NOT contains function with type B
        if question_id == 9:
            class_a = random.randrange(class_id_count)

            current_functions = inputs["classes"][class_a]["functions"]
            for supertype in inputs["classes"][class_a]["superTypes"]:
                current_functions.extend(inputs["classes"][supertype]["functions"])

            chosen_other_function_id = None
            for index, other_description in enumerate(inputs["functions"]):
                chosen = True
                for current_description in current_functions:
                    if functions_are_similar(other_description, current_description):
                        chosen = False
                        break
                if chosen:
                    chosen_other_function_id = index
                    break

            if chosen_other_function_id is None:
                continue

            A = inputs["classes"][class_a]["name"]
            B = function(inputs["classes"], inputs["functions"][chosen_other_function_id])
            text = '%s contains member %s - NO' % (A, B)
            return QuestionSample(question_id, [class_a, chosen_other_function_id], 0.0, text)
    
        # (A, X): A contains parameter with type X
        if question_id == 10:
            functions_with_parameters = []
            for function_id in range(function_id_count):
                if len(inputs["functions"][function_id]["parameters"]) != 0:
                    functions_with_parameters.append(function_id)

            if len(functions_with_parameters) == 0:
                continue

            function_a = random.choice(functions_with_parameters)
            all_parameters = inputs["functions"][function_a]["parameters"]
            parameter_id = random.choice(all_parameters)

            A = function(inputs["classes"], inputs["functions"][function_a])
            X = inputs["classes"][parameter_id]["name"]
            text = '%s contains parameter %s - YES' % (A, X)
            return QuestionSample(question_id, [function_a, parameter_id], 1.0, text)

        # (A, X): A contains parameter with type Y, Y is supertype of X
        if question_id == 11:
            functions_with_parameters = []
            for function_id in range(function_id_count):
                if len(inputs["functions"][function_id]["parameters"]) != 0:
                    functions_with_parameters.append(function_id)

            if len(functions_with_parameters) == 0:
                continue

            chosen_function_id = None
            chosen_parameter_id = None
            for function_id in functions_with_parameters:
                all_parameters = inputs["functions"][function_id]["parameters"]

                all_subtypes = []
                for p in all_parameters:
                    all_subtypes.extend(subtypes_of(p, inputs))
                unused_subtypes = list(filter(lambda p: p not in all_parameters, all_subtypes))

                if len(unused_subtypes) != 0:
                    chosen_function_id = function_id
                    chosen_parameter_id = random.choice(unused_subtypes)

            if chosen_function_id is None:
                continue

            A = function(inputs["classes"], inputs["functions"][chosen_function_id])
            X = inputs["classes"][chosen_parameter_id]["name"]
            text = '%s contains parameter %s - YES' % (A, X)
            return QuestionSample(question_id, [chosen_function_id, chosen_parameter_id], 1.0, text)

        # (A, X): A NOT contains parameter with type X
        if question_id == 12:
            function_a = random.randrange(function_id_count)

            all_parameters = inputs["functions"][function_a]["parameters"]
            all_subtypes = []
            for p in all_parameters:
                all_subtypes.extend(subtypes_of(p, inputs))
            unused_types = list(filter(lambda p: p not in all_parameters and p not in all_subtypes, [i for i in range(class_id_count)]))

            if len(unused_types) == 0:
                continue

            parameter_id = random.choice(unused_types)

            A = function(inputs["classes"], inputs["functions"][function_a])
            X = inputs["classes"][parameter_id]["name"]
            text = '%s contains parameter %s - NO' % (A, X)
            return QuestionSample(question_id, [function_a, parameter_id], 0.0, text)

        # (A, B): A return B
        if question_id == 13:
            function_a = random.randrange(function_id_count)

            return_type = inputs["functions"][function_a]["returnType"]

            A = function(inputs["classes"], inputs["functions"][function_a])
            B = inputs["classes"][return_type]["name"]
            text = '%s returns %s - YES' % (A, B)
            return QuestionSample(question_id, [function_a, return_type], 1.0, text)

        # (A, B): A return C, C is subtype of B
        if question_id == 14:
            functions_where_return_has_supertypes = []
            for function_id in range(function_id_count):
                return_type = inputs["functions"][function_id]["returnType"]
                if len(inputs["classes"][function_id]["superTypes"]) != 0:
                    functions_where_return_has_supertypes.append(function_id)

            if len(functions_where_return_has_supertypes) == 0:
                continue

            function_a = random.choice(functions_where_return_has_supertypes)

            return_type = inputs["functions"][function_a]["returnType"]
            all_supertypes = inputs["classes"][function_a]["superTypes"]
            return_type = random.choice(all_supertypes)

            A = function(inputs["classes"], inputs["functions"][function_a])
            B = inputs["classes"][return_type]["name"]
            text = '%s returns %s - YES' % (A, B)
            return QuestionSample(question_id, [function_a, return_type], 1.0, text)

        # (A, B): A NOT return B
        if question_id == 15:
            function_a = random.randrange(function_id_count)

            return_type = inputs["functions"][function_a]["returnType"]
            all_subtypes = subtypes_of(return_type, inputs)
            unused_types = list(filter(lambda p: p != return_type and p not in all_subtypes, [i for i in range(class_id_count)]))

            if len(unused_types) == 0:
                continue

            return_type = random.choice(unused_types)

            A = function(inputs["classes"], inputs["functions"][function_a])
            B = inputs["classes"][return_type]["name"]
            text = '%s returns %s - NO' % (A, B)
            return QuestionSample(question_id, [function_a, return_type], 0.0, text)


def function(classes, fn):
    params = fn["parameters"]
    returns = fn["returnType"]

    params = list(map(lambda p: classes[p]["name"], params))
    returns = classes[returns]["name"]

    return '(' + ", ".join(params) + ') -> ' + returns
