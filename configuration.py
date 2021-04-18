import pathlib

from os.path import join


class Configuration:
    root_path = pathlib.Path(__file__).parent.absolute()

    integer2string_json = join(root_path, 'dataset', 'integer2string.json')
    string2integer_json = join(root_path, 'dataset', 'string2integer.json')
    train_dataset_json = join(root_path, 'dataset', 'dataset.json')

    saved_model = join(root_path, 'saved_model', 'model')

    print_shape = False

    train_dataset_size = 8000
    test_dataset_size = 993
    test_dataset_begin = 8000
    test_dataset_end = 8993

    train_batch_size = 10
    test_batch_size = 10
    predict_batch_size = 10

    vocabulary_size = 111
    node_embedding_dim = 32
    path_embedding_dim = 128
    encoder_ff_first_layer_dim = 128
    encoder_attention_heads_count = 8  # TODO: из-за этого меняется рамерность выхода энкодера, как быть?
    max_child_index = 15

    epochs_count = 2
