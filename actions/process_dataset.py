import json


def process_dataset():
    with open('/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/integer/integer2string.json', 'r') as file:
        index2word = json.load(file)
        for (k, v) in list(index2word.items()):
            index2word.pop(k)
            index2word[int(k)] = v

    with open('/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/integer/string2integer.json', 'r') as file:
        word2index = json.load(file)

    print(list(index2word.items()))
    print(list(word2index.items())[:5])

    def to_vector(n):
        return [1.0 if n == i else 0.0 for i in range(110)]

    composed = []
    targets = []

    with open('/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/integer/dataset.json', 'r') as file:
        for line in file:
            for sample in json.loads(line):
                leaf_paths = sample["leafPaths"]
                root_path = sample["rootPath"]
                index_among_brothers = sample["indexAmongBrothers"]
                target = sample["target"]

                composed.append(leaf_paths + [root_path])
                targets.append(to_vector(target))

    print(composed[:3])
    print(targets[:3])

    return composed, targets, index2word
