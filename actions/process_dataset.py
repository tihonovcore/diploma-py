import json

from configuration import Configuration
from random import shuffle


class ProcessedDataset:
    def __init__(self, composed, left_brothers, target_indices, targets, integer2string, string2integer):
        self.composed = composed
        self.left_brothers = left_brothers
        self.target_indices = target_indices
        self.targets = targets
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
    target_indices = []
    targets = []
    left_brothers = []

    with open(path_to_dataset_json, 'r') as file:
        for line in file:
            for sample in json.loads(line):
                leaf_paths = sample["leafPaths"]
                root_path = sample["rootPath"]
                brothers = sample["leftBrothers"]
                index_among_brothers = sample["indexAmongBrothers"]
                target = sample["target"]

                composed.append(leaf_paths + [root_path])
                left_brothers.append(brothers)
                target_indices.append(index_among_brothers)
                targets.append(to_vector(target))

    zipped = list(zip(composed, left_brothers, target_indices, targets))
    if shuffle_dataset: shuffle(zipped)
    composed, left_brothers, target_indices, targets = list(zip(*zipped))

    return ProcessedDataset(composed, left_brothers, target_indices, targets, index2word, word2index)
