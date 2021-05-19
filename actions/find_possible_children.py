import json

import tensorflow as tf

from configuration import Configuration


def get_weights_batch(x_batch, left_brothers_batch):
    impossible_result = []
    possible_result = []
    for sample, left_brothers_kind_ids in zip(x_batch, left_brothers_batch):
        root_path = sample[-1]
        parent_kind_id = root_path[-1]

        possible_children = parent_id_to_children_ids(parent_kind_id, left_brothers_kind_ids)
        impossible_children = list(filter(lambda c: c not in possible_children, [i for i in range(Configuration.vocabulary_size)]))

        impossible_result.append(impossible_children)
        possible_result.append(possible_children)

    return possible_result, tf.ragged.constant(impossible_result)


def parent_id_to_children_ids(parent_kind_id, left_brothers_kind_ids):
    parent_kind_id = parent_kind_id.numpy()

    with open(Configuration.parent_child_json, 'r') as file:
        parent2child = json.loads(file.read())

    if len(left_brothers_kind_ids) != 0 and left_brothers_kind_ids[-1] == Configuration.string2integer["AFTER_LAST"]:
        return []

    if parent_kind_id not in list(map(lambda s: Configuration.string2integer[s], ["BODY", "CLASS_BODY", "BLOCK", "FILE", "VALUE_PARAMETER_LIST", "STRING_TEMPLATE", "WHEN", "IMPORT_LIST", "VALUE_ARGUMENT_LIST", "TYPE_ARGUMENT_LIST", "TYPE_PARAMETER_LIST"])):
        context = [parent_kind_id]
        context.extend(left_brothers_kind_ids.numpy())
        string_context = list(map(lambda integer: Configuration.integer2string[integer], context))
        string_context = ", ".join(string_context)
    else:
        string_context = Configuration.integer2string[parent_kind_id]

    if string_context in parent2child:
        children_strings = parent2child[string_context]
    else:
        children_strings = []

    return list(map(lambda child: Configuration.string2integer[child], children_strings))
