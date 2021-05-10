import json
from os.path import join

from configuration import Configuration
from os import walk
from random import shuffle


class ProcessedDataset:
    def __init__(self, composed, left_brothers, target_indices, targets, type_container_id, json_type_containers, integer2string, string2integer):
        self.composed = composed
        self.left_brothers = left_brothers
        self.target_indices = target_indices
        self.targets = targets
        self.type_container_id = type_container_id
        self.json_type_containers = json_type_containers
        self.integer2string = integer2string
        self.string2integer = string2integer


def process_dataset(path_to_dataset_json=Configuration.train_dataset_json, shuffle_dataset=False):
    with open(Configuration.integer2string_json, 'r') as file:
        index2word = json.load(file)
        for (k, v) in list(index2word.items()):
            index2word.pop(k)
            index2word[int(k)] = v

    with open(Configuration.string2integer_json, 'r') as file:
        word2index = json.load(file)

    assert Configuration.vocabulary_size == len(index2word)
    assert Configuration.vocabulary_size == len(word2index)

    Configuration.integer2string = index2word
    Configuration.string2integer = word2index

    def to_vector(n):
        return [1.0 if n == i else 0.0 for i in range(Configuration.vocabulary_size)]

    composed = []
    left_brothers = []
    target_indices = []
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
                index_among_brothers = parsed_sample["indexAmongBrothers"]
                target = parsed_sample["target"]

                composed.append(leaf_paths + [root_path])
                left_brothers.append(brothers)
                target_indices.append(index_among_brothers)
                targets.append(to_vector(target))
                type_container_id.append(len(json_type_containers))

        with open(path_to_types, 'r') as json_types:
            json_type_containers.append(json_types.read())

    zipped = list(zip(composed, left_brothers, target_indices, targets, type_container_id))
    if shuffle_dataset: shuffle(zipped)
    composed, left_brothers, target_indices, targets, type_container_id = list(zip(*zipped))

    return ProcessedDataset(composed, left_brothers, target_indices, targets, type_container_id, json_type_containers, index2word, word2index)
