import getopt
import sys

from actions.evaluate_statistics import evaluate_statistics
from actions.process_dataset import process_dataset
from configuration import Configuration
from path_model.slm import SLM

if __name__ == '__main__':
    try:
        opts, _ = getopt.getopt(sys.argv[1:], '', ['json_path='])
        path_to_sample = opts[0][1]
    except getopt.GetoptError:
        print('evaluate.py --json_path="/absolute/path/to/sample.json"')
        sys.exit(2)

    processed_dataset = process_dataset(path_to_sample)

    slm = SLM(batch_size=20)  # todo: wtf is batch_size?
    slm.load_weights(Configuration.saved_model)

    evaluate_statistics(
        0,
        6434,
        processed_dataset,
        slm
    )
