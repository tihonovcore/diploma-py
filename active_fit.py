import json
import random
import re
import subprocess
import tensorflow as tf

from os import walk
from os.path import join
from random import shuffle

from actions.find_possible_children import get_weights_batch
from actions.train_model import syntax_loss
from configuration import Configuration
from implementation.slm import SLM
from type_embeddings.question_model import QuestionModel


def must_be_skipped(path):
    if path[-2:] != 'kt':
        return True

    if path.endswith('kt30402.kt') or path.endswith('crossTypeEquals.kt') or path.endswith('jsNative.kt'):
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

    question_model = QuestionModel(mode=Configuration.recurrent_mode)
    question_model.trainable = False
    question_model.load_weights(Configuration.saved_type_model)
    type_embeddings = question_model.type_embeddings

    slm = SLM(batch_size=20)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    shuffle(file_paths)

    for file_path in file_paths:
        with open(Configuration.cooperative__send, 'w') as request:
            request.write(file_path)

        _ = subprocess.run(Configuration.gradle_extract_paths, capture_output=True, shell=True)

        with open(Configuration.cooperative__take) as response_from_kotlin:
            status = response_from_kotlin.read()

        with tf.GradientTape() as tape:
            all_predicted_kinds = []
            all_syntax_losses = []

            while status == "PATH":
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
    
                possible_children, impossible_children = get_weights_batch(composed, left_brothers)
                possible_children = possible_children[0]  # single element at batch

                reconstructed_kind, reconstructed_type = slm((composed, index_among_brothers, type_container_id, leaf_types, root_types, type_container_embeddings))

                syntax_ls = syntax_loss(None, reconstructed_kind, impossible_children)
                all_syntax_losses.append(syntax_ls)

                reconstructed_kind = tf.reshape(reconstructed_kind, (Configuration.vocabulary_size,))  # single element at batch
                with open(Configuration.cooperative__send, 'w') as send:
                    kind_id_among_possible = random.randrange(len(possible_children))
                    kind_str = Configuration.integer2string[possible_children[kind_id_among_possible]]
                    print('%s from %d' % (kind_str, len(possible_children)))

                    kind = tf.gather(reconstructed_kind, possible_children)[kind_id_among_possible]
                    all_predicted_kinds.append(kind)

                    request = '{ "kind": "%s", "type": %d }' % (kind_str, 0)
                    send.write(request)

                _ = subprocess.run(Configuration.gradle_on_predict, capture_output=True, shell=True)

                with open(Configuration.cooperative__take, 'r') as response_from_kotlin:
                    status = response_from_kotlin.read()

            full_loss = tf.constant(0.0)
            for ls in all_syntax_losses:
                full_loss = full_loss + ls

            for prob in all_predicted_kinds:
                if status == "SUCC":
                    full_loss = full_loss - tf.math.log(prob)
                elif status == "FAIL":
                    full_loss = full_loss - tf.math.log(1.0 - prob)

        if status == "SUCC":
            grads = tape.gradient(full_loss, slm.trainable_weights)
            optimizer.apply_gradients(zip(grads, slm.trainable_weights))

            print("last loss = %.4f" % full_loss)
        elif status == "FAIL":
            grads = tape.gradient(full_loss, slm.trainable_weights)
            optimizer.apply_gradients(zip(grads, slm.trainable_weights))

            print("last loss = %.4f" % full_loss)

        if status.startswith("ERROR"):
            print(status)
            break
