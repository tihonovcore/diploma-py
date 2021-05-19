import json
import re
import subprocess
import tensorflow as tf

from os import walk
from os.path import join
from random import shuffle

from actions.find_possible_children import get_weights_batch
from configuration import Configuration
from implementation.slm import SLM
from type_embeddings.question_model import QuestionModel


def must_be_skipped(path):
    if path[-2:] != 'kt':
        return True

    if path.endswith('kt30402.kt') or path.endswith('crossTypeEquals.kt'):
        return True

    with open(path) as source:
        text = source.read()

        if re.search('//\\s*?FILE:', text) is not None:
            return True
        if re.search('//\\s*?WITH_RUNTIME', text) is not None:
            return True
        if re.search('//\\s*?FILE: .*?\\.java', text) is not None:
            return True


if __name__ == '__main__':
    file_paths = []

    for root, _, files in walk(Configuration.kotlin_test_directory):
        for file in files:
            file_path = join(root, file)
            if must_be_skipped(file_path):
                continue

            file_paths.append(file_path)

    question_model = QuestionModel()
    question_model.trainable = False
    question_model.load_weights(Configuration.saved_type_model)
    type_embeddings = question_model.type_embeddings

    slm = SLM(batch_size=20)

    shuffle(file_paths)

    # TODO: use batches?
    for file_path in file_paths[:1]:
        with open(Configuration.cooperative__send, 'w') as request:
            request.write(file_path)

        _ = subprocess.run(Configuration.gradle_extract_paths, capture_output=True, shell=True)

        with open(Configuration.cooperative__take) as response_from_kotlin:
            status = response_from_kotlin.read()

        while status != "SUCC" and status != "FAIL":
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

            reconstructed = slm((composed, index_among_brothers, type_container_id, leaf_types, root_types, type_container_embeddings))

            impossible_children_batch = get_weights_batch(composed, left_brothers)

        #     './gradlew :idea:test --tests "org.jetbrains.kotlin.idea.caches.resolve.OnPredict.testTTT" -q'

        if status == "SUCC":
            pass  # good loss
        elif status == "FAIL":
            pass  # bad loss

# TODO: can we save type embeddings? they may change, but what if not???
