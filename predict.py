import getopt
import sys

from actions.evaluate_statistics import evaluate_statistics
from actions.process_dataset import process_dataset
from configuration import Configuration
from implementation.slm import SLM

if __name__ == '__main__':
    try:
        opts, _ = getopt.getopt(sys.argv[1:], '', ['json_path='])
        path_to_sample = opts[0][1]
    except getopt.GetoptError:
        print('predict.py --json_path="path/to/sample.json"')
        sys.exit(2)

    composed, targets, integer2string = process_dataset(path_to_sample)

    slm = SLM(vocab_size=110, embedding_dim=32, batch_size=20, rnn_units=64, ff_dim=64)
    slm.load_weights(Configuration.saved_model)

    evaluate_statistics(TEST_BEGIN=1500, TEST_END=1600, composed=composed, targets=targets, slm=slm, index2word=integer2string)
