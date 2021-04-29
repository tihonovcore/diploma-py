import json

from actions.evaluate_statistics import parent_id_to_children_ids
from configuration import Configuration
from random import shuffle


class ProcessedDataset:
    def __init__(self, composed, target_indices, targets, loss_weights, integer2string, string2integer):
        self.composed = composed
        self.target_indices = target_indices
        self.targets = targets
        self.loss_weights = loss_weights
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

    def to_vector(n):
        return [1.0 if n == i else 0.0 for i in range(Configuration.vocabulary_size)]

    composed = []
    target_indices = []
    targets = []
    loss_weights = []

    with open(path_to_dataset_json, 'r') as file:
        for line in file:
            for sample in json.loads(line):
                leaf_paths = sample["leafPaths"]
                root_path = sample["rootPath"]
                index_among_brothers = sample["indexAmongBrothers"]
                target = sample["target"]

                parent = root_path[-1]
                possible_children = parent_id_to_children_ids(parent, index2word)
                weights = [1.0 if i in possible_children else Configuration.loss_alpha for i in range(Configuration.vocabulary_size)]
                weights[target] = Configuration.loss_alpha

                composed.append(leaf_paths + [root_path])
                target_indices.append(index_among_brothers)
                targets.append(to_vector(target))
                loss_weights.append(weights)

    dataset_size = Configuration.train_dataset_size + Configuration.test_dataset_size
    assert len(composed) == dataset_size
    assert len(target_indices) == dataset_size
    assert len(targets) == dataset_size
    assert len(loss_weights) == dataset_size

    zipped = list(zip(composed, target_indices, targets, loss_weights))
    if shuffle_dataset: shuffle(zipped)
    composed, target_indices, targets, loss_weights = list(zip(*zipped))

    return ProcessedDataset(composed, target_indices, targets, loss_weights, index2word, word2index)
