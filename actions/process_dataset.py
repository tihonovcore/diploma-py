import json
from os.path import join

from configuration import Configuration
from os import walk
from random import shuffle

from type_model.question_model import QuestionModel


class ProcessedDataset:
    def __init__(self, composed, left_brothers, target_indices, targets, type_container_id, types_for_leaf_paths, types_for_root_path, type_container_embeddings, integer2string, string2integer):
        self.composed = composed
        self.left_brothers = left_brothers
        self.target_indices = target_indices
        self.targets = targets
        self.type_container_id = type_container_id
        self.types_for_leaf_paths = types_for_leaf_paths
        self.types_for_root_path = types_for_root_path
        self.type_container_embeddings = type_container_embeddings
        self.integer2string = integer2string
        self.string2integer = string2integer


def process_dataset(path_to_dataset_json=Configuration.train_dataset_json, shuffle_dataset=False):
    def to_vector(n):
        return [1.0 if n == i else 0.0 for i in range(Configuration.vocabulary_size)]

    composed = []
    left_brothers = []
    target_indices = []
    types_for_leaf_paths = []
    types_for_root_path = []
    targets = []
    json_type_containers = []
    type_container_id = []

    _, dirnames, _ = next(walk(Configuration.new_train_dataset_json))
    for dirname in dirnames:
        path_to_samples = join(Configuration.new_train_dataset_json, dirname, 'paths')
        path_to_types = join(Configuration.new_train_dataset_json, dirname, 'types.json')

        _, _, samples = next(walk(path_to_samples))
        for sample in samples:
            with open(join(path_to_samples, sample), 'r') as json_sample:
                parsed_sample = json.loads(json_sample.read())

                leaf_paths = parsed_sample["leafPaths"]
                root_path = parsed_sample["rootPath"]
                brothers = parsed_sample["leftBrothers"]
                leaf_types = parsed_sample["typesForLeafPaths"]
                root_types = parsed_sample["typesForRootPath"]
                index_among_brothers = parsed_sample["indexAmongBrothers"]
                target = parsed_sample["target"]

                composed.append(leaf_paths + [root_path])
                left_brothers.append(brothers)
                types_for_leaf_paths.append(leaf_types)
                types_for_root_path.append(root_types)
                target_indices.append(index_among_brothers)
                targets.append(to_vector(target))
                type_container_id.append(len(json_type_containers))

        model = QuestionModel()
        model.trainable = False
        model.load_weights(Configuration.saved_type_model)
        type_embeddings = model.type_embeddings

        with open(path_to_types, 'r') as json_types:
            container = json.loads(json_types.read())
            class_embeddings, _, _ = type_embeddings(container)
            json_type_containers.append(class_embeddings)

    zipped = list(zip(composed, left_brothers, target_indices, targets, type_container_id, types_for_leaf_paths, types_for_root_path))
    if shuffle_dataset: shuffle(zipped)
    composed, left_brothers, target_indices, targets, type_container_id, types_for_leaf_paths, types_for_root_path = list(zip(*zipped))

    return ProcessedDataset(composed, left_brothers, target_indices, targets, type_container_id, types_for_leaf_paths, types_for_root_path, json_type_containers, Configuration.integer2string, Configuration.string2integer)
