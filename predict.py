import getopt
import sys
import random

import tensorflow as tf

from actions.evaluate_statistics import parent_id_to_children_ids
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

    composed, target_indices, targets, integer2string = process_dataset(path_to_sample)

    slm = SLM(batch_size=20)  # todo: wtf is batch_size?
    slm.load_weights(Configuration.saved_model)

    result = []

    batch_size = Configuration.predict_batch_size
    for begin in range(0, len(composed), batch_size):
        composed_batch = tf.ragged.constant(composed[begin:begin + batch_size])
        indices_batch = tf.constant(target_indices[begin:begin + batch_size])

        batch_result = slm.call((composed_batch, indices_batch))

        for (res, cmp) in zip(batch_result, composed_batch):
            parent_id = cmp[-1][-1].numpy()
            children_ids = parent_id_to_children_ids(parent_id, integer2string)

            _, predicted = tf.nn.top_k(tf.gather(res, children_ids), k=min(5, len(children_ids)))
            random_choice = random.choice(list(map(lambda pred: children_ids[pred], predicted.numpy())))
            random_choice = integer2string[random_choice]
            result.append(random_choice)

    for r in result:
        print(r)
