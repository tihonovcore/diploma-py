import json
import tensorflow as tf

from active_fit.fit_environment.FitEnvironment import FitEnvironment
from active_fit.fit_environment.TypedFitEnvironment import TypedFitEnvironment
from active_fit.prepared_data.PreparedData import PreparedData
from active_fit.prepared_data.TypedPreparedData import TypedPreparedData
from configuration import Configuration
from functools import singledispatch


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

    leaf_paths = paths_info["leafPaths"]
    root_path = paths_info["rootPath"]
    left_brothers = paths_info["leftBrothers"]
    leaf_types = paths_info["typesForLeafPaths"]
    root_types = paths_info["typesForRootPath"]
    index_among_brothers = paths_info["indexAmongBrothers"]

    class_embeddings, _, _ = env.type_embeddings(types_info)

    composed = leaf_paths + [root_path]
    type_container_id = [0]  # there is single container
    type_container_embeddings = [class_embeddings]
    leaf_types = [leaf_types]
    root_types = [root_types]

    composed = tf.ragged.constant([composed], dtype='float32')
    left_brothers = tf.ragged.constant([left_brothers])
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
