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

    composed, targets, integer2string = process_dataset(path_to_sample)

    slm = SLM(vocab_size=111, embedding_dim=32, batch_size=20, rnn_units=64, ff_dim=64)
    slm.load_weights(Configuration.saved_model)

    result = []

    step = 10
    for begin in range(0, len(composed), step):
        batch_result = slm.call(tf.ragged.constant(composed[begin:begin + step]))
        batch_result = tf.argmax(batch_result, axis=1).numpy()
        batch_result = list(map(lambda t: integer2string[t], batch_result))
        result.extend(batch_result)

    for r in result:
        print(r)
