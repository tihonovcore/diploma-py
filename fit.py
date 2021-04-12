from actions.process_dataset import process_dataset
from actions.train_model import train_model
from actions.evaluate_statistics import evaluate_statistics
from configuration import Configuration

if __name__ == '__main__':
    composed, targets, integer2string = process_dataset()
    slm = train_model(composed, targets, USE_N_SAMPLES=8000)
    evaluate_statistics(TEST_BEGIN=8000, TEST_END=9000, composed=composed, targets=targets, slm=slm, index2word=integer2string)

    slm.save_weights(Configuration.saved_model)
