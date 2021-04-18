from actions.process_dataset import process_dataset
from actions.train_model import train_model
from actions.evaluate_statistics import evaluate_statistics
from configuration import Configuration

if __name__ == '__main__':
    composed, targets, integer2string = process_dataset()
    slm = train_model(composed, targets)
    evaluate_statistics(
        Configuration.test_dataset_begin,
        Configuration.test_dataset_end,
        composed,
        targets,
        slm,
        integer2string
    )

    slm.save_weights(Configuration.saved_model)
