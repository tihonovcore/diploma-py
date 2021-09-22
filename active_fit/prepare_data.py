import json
import tensorflow as tf

from active_fit.fit_environment.FitEnvironment import FitEnvironment
from active_fit.fit_environment.TypedFitEnvironment import TypedFitEnvironment
from active_fit.prepared_data.PreparedData import PreparedData
from active_fit.prepared_data.TypedPreparedData import TypedPreparedData
from configuration import Configuration
from functools import singledispatch
from tensorflow import RaggedTensor
from tensorflow.python.framework.ops import EagerTensor
from typing import List, Dict


@singledispatch
def prepare_data(env: FitEnvironment):
    with open(Configuration.cooperative__paths, 'r') as json_paths:
        paths_info = json.load(json_paths)

    leaf_paths = paths_info["leafPaths"]
    root_path = paths_info["rootPath"]
    left_brothers = paths_info["leftBrothers"]
    index_among_brothers = paths_info["indexAmongBrothers"]

    composed = leaf_paths + [root_path]
    composed = tf.ragged.constant([composed], dtype='float32')
    left_brothers = tf.ragged.constant([left_brothers])
    index_among_brothers = tf.constant([index_among_brothers])

    return PreparedData(
        composed,
        left_brothers,
        index_among_brothers
    )


@prepare_data.register
def _(env: TypedFitEnvironment):
    with open(Configuration.cooperative__paths, 'r') as json_paths:
        paths_info = json.load(json_paths)
    with open(Configuration.cooperative__types, 'r') as json_types:
        types_info = json.load(json_types)

    leaf_paths: List[List[int]] = paths_info["leafPaths"]
    root_path: List[int] = paths_info["rootPath"]
    left_brothers: List[int] = paths_info["leftBrothers"]
    leaf_types: List[Dict[str, int]] = paths_info["typesForLeafPaths"]
    root_types: Dict[str, int] = paths_info["typesForRootPath"]
    index_among_brothers: int = paths_info["indexAmongBrothers"]

    class_embeddings, _, _ = env.type_embeddings(types_info)

    composed: List[List[int]] = leaf_paths + [root_path]
    type_container_id: List[int] = [0]  # there is single container
    type_container_embeddings: List[EagerTensor] = [class_embeddings]
    leaf_types: List[List[Dict[str, int]]] = [leaf_types]
    root_types: List[Dict[str, int]] = [root_types]

    composed: RaggedTensor = tf.ragged.constant([composed], dtype='float32')
    left_brothers: RaggedTensor = tf.ragged.constant([left_brothers])
    index_among_brothers = tf.constant([index_among_brothers])

    return TypedPreparedData(
        composed,
        left_brothers,
        type_container_id,
        type_container_embeddings,
        leaf_types,
        root_types,
        index_among_brothers
    )
