import random
import tensorflow as tf

from actions.find_possible_children import get_weights_batch
from active_fit.loss import TreeGenerationLoss
from active_fit.prepared_data.PreparedData import PreparedData
from active_fit.prepared_data.TypedPreparedData import TypedPreparedData
from configuration import Configuration
from functools import singledispatch
from path_model.abstract_slm import SLM
from tensorflow import RaggedTensor
from tensorflow.python.framework.ops import EagerTensor
from typing import List


@singledispatch
def predict(prepared_data, slm: SLM, loss: TreeGenerationLoss, request: List[str], depth: int = 0) -> int:
    raise Exception('not default implementation')


@predict.register
def _(prepared_data: PreparedData, slm: SLM, loss: TreeGenerationLoss, request: List[str], depth: int = 0) -> int:
    possible_children, impossible_children = get_weights_batch(prepared_data.composed, prepared_data.left_brothers)
    possible_children = possible_children[0]  # single element at batch

    _index_among_brothers = tf.constant(prepared_data.left_brothers.shape[0], shape=(1,))
    reconstructed_kind = slm(
        (
            prepared_data.composed,
            _index_among_brothers
        )
    )

    syntax_ls: EagerTensor = loss.syntax_loss(None, reconstructed_kind, impossible_children)
    loss.all_syntax_losses.append(syntax_ls)

    reconstructed_kind: EagerTensor = tf.reshape(reconstructed_kind, (Configuration.vocabulary_size,))  # single element at batch

    # todo: make probability less
    al_probability: float = 1.0 * max(depth, prepared_data.left_brothers.shape[0])

    if Configuration.string2integer['AFTER_LAST'] in possible_children and random.random() < al_probability:
        kind_id: int = Configuration.string2integer['AFTER_LAST']
    else:
        kind_id_among_possible: int = random.randrange(len(possible_children))
        kind_id: int = possible_children[kind_id_among_possible]

    kind_str: str = Configuration.integer2string[kind_id]
    print('%s from %d' % (kind_str, len(possible_children)))

    kind_: EagerTensor = reconstructed_kind[kind_id]
    loss.all_predicted_kinds.append(kind_)

    request.append('{ "kind": "%s", "type": %d }' % (kind_str, 0))

    if kind_str == 'AFTER_LAST':
        return kind_id

    composed: RaggedTensor = update_paths(prepared_data.composed, kind_id)
    predicted_children: List[int] = []

    while True:
        prediction = predict(
            prepared_data.updated(composed, tf.ragged.constant([predicted_children])), slm, loss, request, depth + 1
        )

        predicted_children.append(prediction)
        if prediction == Configuration.string2integer['AFTER_LAST']:
            break

    return kind_id


@predict.register
def _(prepared_data: TypedPreparedData, slm: SLM, loss: TreeGenerationLoss, request: List[str], depth: int = 0) -> int:
    possible_children, impossible_children = get_weights_batch(prepared_data.composed, prepared_data.left_brothers)
    possible_children = possible_children[0]  # single element at batch

    _index_among_brothers = tf.constant(prepared_data.left_brothers.shape[0], shape=(1,))
    reconstructed_kind, reconstructed_type = slm(
        (
            prepared_data.composed,
            _index_among_brothers,
            prepared_data.type_container_id,
            prepared_data.leaf_types,
            prepared_data.root_types,
            prepared_data.type_container_embeddings
        )
    )

    syntax_ls: EagerTensor = loss.syntax_loss(None, reconstructed_kind, impossible_children)
    loss.all_syntax_losses.append(syntax_ls)

    reconstructed_kind: EagerTensor = tf.reshape(reconstructed_kind, (Configuration.vocabulary_size,))  # single element at batch
    reconstructed_type: EagerTensor = reconstructed_type[0]  # single element at batch

    # todo: make probability less
    al_probability: float = 1.0 * max(depth, prepared_data.left_brothers.shape[0])

    if Configuration.string2integer['AFTER_LAST'] in possible_children and random.random() < al_probability:
        kind_id: int = Configuration.string2integer['AFTER_LAST']
    else:
        kind_id_among_possible: int = random.randrange(len(possible_children))
        kind_id: int = possible_children[kind_id_among_possible]

    kind_str: str = Configuration.integer2string[kind_id]
    print('%s from %d' % (kind_str, len(possible_children)))

    kind_: EagerTensor = reconstructed_kind[kind_id]
    loss.all_predicted_kinds.append(kind_)

    type_id: int = tf.argmax(reconstructed_type).numpy()
    type_: EagerTensor = reconstructed_type[type_id]

    if kind_str != 'AFTER_LAST':
        loss.all_predicted_types.append(type_)

    request.append('{ "kind": "%s", "type": %d }' % (kind_str, type_id))

    if kind_str == 'AFTER_LAST':
        return kind_id

    composed: RaggedTensor = update_paths(prepared_data.composed, kind_id)
    predicted_children: List[int] = []

    while True:
        prediction = predict(
            prepared_data.updated(composed, tf.ragged.constant([predicted_children])), slm, loss, request, depth + 1
        )

        predicted_children.append(prediction)
        if prediction == Configuration.string2integer['AFTER_LAST']:
            break

    return kind_id


def update_paths(old_composed, new_kind) -> RaggedTensor:
    addition: List[int] = [Configuration.string2integer['↓'], new_kind]

    def up_path(x):
        return tf.concat([x, addition], axis=0)

    def up_batch_elem(x):
        return tf.map_fn(up_path, x)

    return tf.map_fn(up_batch_elem, old_composed)
