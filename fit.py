from actions.process_dataset import process_dataset
from actions.train_model import train_model
from actions.evaluate_statistics import evaluate_statistics

if __name__ == '__main__':
    composed, targets, integer2string = process_dataset()
    slm = train_model(composed, targets, USE_N_SAMPLES=20)
    evaluate_statistics(TEST_BEGIN=0, TEST_END=20, composed=composed, targets=targets, slm=slm, index2word=integer2string)
