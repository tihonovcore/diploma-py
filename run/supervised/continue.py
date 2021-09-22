from actions.process_dataset import process_dataset
from actions.train_model import train_model
from actions.evaluate_statistics import evaluate_statistics
from configuration import Configuration
from path_model.slm import SLM

if __name__ == '__main__':
    processed_dataset = process_dataset()

    slm = SLM(batch_size=20)  # todo: wtf is batch_size?
    slm.load_weights(Configuration.saved_model)
    slm = train_model(processed_dataset, slm)
    evaluate_statistics(
        Configuration.test_dataset_begin,
        Configuration.test_dataset_end,
        processed_dataset,
        slm
    )

    slm.save_weights(Configuration.saved_model)
