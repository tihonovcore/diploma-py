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

    result = slm.call(tf.ragged.constant(composed))
    result = tf.argmax(result, axis=1).numpy()
    result = list(map(lambda t: integer2string[t], result))
    for r in result:
        print(r)
