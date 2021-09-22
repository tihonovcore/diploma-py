import json
import tensorflow as tf

from configuration import Configuration
from type_model.implementation import TE


class PreparedData:
    def __init__(
            self,
            composed,
            left_brothers,
            type_container_id,
            type_container_embeddings,
            leaf_types,
            root_types,
            index_among_brothers
    ):
        self.composed = composed
        self.left_brothers = left_brothers
        self.type_container_id = type_container_id
        self.type_container_embeddings = type_container_embeddings
        self.leaf_types = leaf_types
        self.root_types = root_types
        self.index_among_brothers = index_among_brothers

    def updated(self, new_composed, new_left_brothers):
        return PreparedData(
            new_composed,
            new_left_brothers,
            self.type_container_id,
            self.type_container_embeddings,
            self.leaf_types,
            self.root_types,
            self.index_among_brothers
        )


def prepare_data(type_embeddings: TE) -> PreparedData:
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

    class_embeddings, _, _ = type_embeddings(types_info)

    composed = leaf_paths + [root_path]
    type_container_id = [0]  # there is single container
    type_container_embeddings = [class_embeddings]
    leaf_types = [leaf_types]
    root_types = [root_types]

    composed = tf.ragged.constant([composed], dtype='float32')
    left_brothers = tf.ragged.constant([left_brothers])
    index_among_brothers = tf.constant([index_among_brothers])

    return PreparedData(
        composed,
        left_brothers,
        type_container_id,
        type_container_embeddings,
        leaf_types,
        root_types,
        index_among_brothers
    )
