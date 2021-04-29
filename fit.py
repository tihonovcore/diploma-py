from actions.process_dataset import process_dataset
from actions.train_model import train_model
from actions.evaluate_statistics import evaluate_statistics
from configuration import Configuration

if __name__ == '__main__':
    processed_dataset = process_dataset(shuffle_dataset=True)
    slm = train_model(processed_dataset)
    evaluate_statistics(
        Configuration.test_dataset_begin,
        Configuration.test_dataset_end,
        processed_dataset,
        slm
    )

    slm.save_weights(Configuration.saved_model)
