import getopt
import sys
import random

import tensorflow as tf

from actions.find_possible_children import parent_id_to_children_ids
from actions.process_dataset import process_dataset
from configuration import Configuration
from implementation.slm import SLM

if __name__ == '__main__':
    try:
        opts, _ = getopt.getopt(sys.argv[1:], '', ['json_path='])
        path_to_sample = opts[0][1]
    except getopt.GetoptError:
        print('predict.py --json_path="/absolute/path/to/sample.json"')
        sys.exit(2)

    processed_dataset = process_dataset(path_to_sample)

    slm = SLM(batch_size=20)  # todo: wtf is batch_size?
    slm.load_weights(Configuration.saved_model)

    result = []

    batch_size = Configuration.predict_batch_size
    for begin in range(0, len(processed_dataset.composed), batch_size):
        composed_batch = tf.ragged.constant(processed_dataset.composed[begin:begin + batch_size])
        left_brothers_batch = tf.ragged.constant(processed_dataset.left_brothers[begin:begin + batch_size])
        leaf_types = processed_dataset.types_for_leaf_paths[begin:begin + batch_size]
        root_types = processed_dataset.types_for_root_path[begin:begin + batch_size]
        indices_batch = tf.constant(processed_dataset.target_indices[begin:begin + batch_size])
        type_container_id_batch = processed_dataset.type_container_id[begin:begin + batch_size]

        batch_result = slm.call((composed_batch, indices_batch, type_container_id_batch, leaf_types, root_types, processed_dataset.type_container_embeddings))

        for (res, cmp, left_brothers) in zip(batch_result, composed_batch, left_brothers_batch):
            parent_id = cmp[-1][-1].numpy()
            children_ids = parent_id_to_children_ids(parent_id, left_brothers)

            _, predicted = tf.nn.top_k(tf.gather(res, children_ids), k=min(5, len(children_ids)))
            random_choice = random.choice(list(map(lambda pred: children_ids[pred], predicted.numpy())))
            random_choice = processed_dataset.integer2string[random_choice]
            result.append(random_choice)

    for r in result:
        print(r)
