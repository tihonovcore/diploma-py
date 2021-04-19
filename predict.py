import getopt
import sys
import tensorflow as tf

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
        batch_result = tf.argmax(batch_result, axis=1).numpy()
        batch_result = list(map(lambda t: integer2string[t], batch_result))

        result.extend(batch_result)

    for r in result:
        print(r)
