import json

from configuration import Configuration


def process_dataset(path_to_dataset_json=Configuration.train_dataset_json):
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

    with open(path_to_dataset_json, 'r') as file:
        for line in file:
            for sample in json.loads(line):
                leaf_paths = sample["leafPaths"]
                root_path = sample["rootPath"]
                index_among_brothers = sample["indexAmongBrothers"]
                target = sample["target"]

                composed.append(leaf_paths + [root_path])
                target_indices.append(index_among_brothers)
                targets.append(to_vector(target))

    # todo: shuffle dataset
    return composed, target_indices, targets, index2word
