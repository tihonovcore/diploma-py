import pathlib

from os.path import join


class Configuration:
    root_path = pathlib.Path(__file__).parent.absolute()

    integer2string = None
    string2integer = None

    parent_child_json = join(root_path, 'dataset', 'parentChild.json')
    integer2string_json = join(root_path, 'dataset', 'integer2string.json')
    string2integer_json = join(root_path, 'dataset', 'string2integer.json')
    train_dataset_json = join(root_path, 'dataset', 'dataset.json')
    new_train_dataset_json = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/integer'

    saved_model = join(root_path, 'saved_model', 'model')

    print_shape = False

    train_dataset_size = 6000
    test_dataset_size = 434
    test_dataset_begin = 6000
    test_dataset_end = 6434

    train_batch_size = 10
    test_batch_size = 10
    predict_batch_size = 10

    vocabulary_size = 111
    node_embedding_dim = 16
    path_embedding_dim = 128
    encoder_ff_first_layer_dim = 128
    encoder_attention_heads_count = 8
    max_child_index = 16  # [0; 15]

    loss_alpha = 2.0

    epochs_count = 2

    # Question model configuration:

    saved_type_model = join(root_path, 'type_embeddings', 'weights', 'weights')

    type_embedding_model_epochs_count = 10

    basic_types_count = 11
    type_embedding_dim = 96

    question_type_count = 16
    questions_per_file_train = 5
    questions_per_file_test = 2

    types_dataset = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/types'

    kotlin_test_directory = '/home/tihonovcore/diploma/kotlin/compiler/testData/codegen/box'
    cooperative__send = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/request.txt'
    cooperative__take = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/answer.txt'
    cooperative__paths = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/paths.json'
    cooperative__types = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/types.json'

    cd = 'cd /home/tihonovcore/diploma/kotlin; '
    gradle_extract_paths = cd + './gradlew :idea:test --tests "org.jetbrains.kotlin.idea.caches.resolve.ExtractPaths.testTTT" -q'
    gradle_on_predict = cd + './gradlew :idea:test --tests "org.jetbrains.kotlin.idea.caches.resolve.OnPredict.testTTT" -q'
