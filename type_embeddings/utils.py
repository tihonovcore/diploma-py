import random


def subtypes_of(class_id, inputs):
    subtypes = []
    for klass_id, klass in enumerate(inputs["classes"]):
        if class_id in klass["superTypes"]:
            subtypes.append(klass_id)
    return subtypes


def get_all_properties(class_id, inputs):
    result = []

    def walk_through_supertypes(current):
        result.extend(current["properties"])
        for supertype in current["superTypes"]:
            walk_through_supertypes(inputs["classes"][supertype])

    walk_through_supertypes(inputs["classes"][class_id])
    return result


def randomly_change_to_subtype(property_ids, inputs):
    for i in range(len(property_ids)):
        if random.random() < 0.1:
            property_ids[i] = random.choice(subtypes_of(property_ids[i], inputs))
    return property_ids


def functions_are_similar(a, b):
    if a["returnType"] != b["returnType"]:
        return False

    a_params = a["parameters"]
    b_params = b["parameters"]
    if len(a_params) != len(b_params):
        return False

    return sorted(a_params) == sorted(b_params)
