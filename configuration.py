import json
import pathlib

from os.path import join

from ModelMode import ModelMode


class Configuration:
    model_mode: ModelMode = ModelMode.TYPED__INJECTION_VIA_NODES

    root_path: str = pathlib.Path(__file__).parent.absolute()

    parent_child_json = join(root_path, 'dataset', 'parentChild.json')
    integer2string_json = join(root_path, 'dataset', 'integer2string.json')
    string2integer_json = join(root_path, 'dataset', 'string2integer.json')
    train_dataset_json = join(root_path, 'dataset', 'dataset.json')
    new_train_dataset_json = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/integer'

    saved_model: str = join(root_path, 'path_model', model_mode.to_string(), 'weights')

    print_shape: bool = False

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

    with open(integer2string_json, 'r') as file:
        integer2string = json.load(file)
        for (k, v) in list(integer2string.items()):
            integer2string.pop(k)
            integer2string[int(k)] = v

    with open(string2integer_json, 'r') as file:
        string2integer = json.load(file)

    assert vocabulary_size == len(integer2string)
    assert vocabulary_size == len(string2integer)

    # Question model configuration:

    recurrent_mode = 'gru'

    saved_type_model = join(root_path, 'type_model', recurrent_mode, 'weights')

    type_embedding_model_epochs_count = 1

    basic_types_count = 11
    type_embedding_dim = 16

    question_type_count = 15
    questions_per_file_train = 200

    # 0 - don't use types                       # todo: problems with dimensions
    # 1 - old inject type into node embedding   # todo: remove (?)
    # 2 - inject type into node embedding       # note: too slow and memory wasted (?)
    # 3 - inject type via context               # note: not yet implemented
    type_injection_approach = 2

    types_dataset = join(root_path, 'dataset', 'types_dataset')

    kotlin_test_directory: str = '/home/tihonovcore/diploma/kotlin/compiler/testData/codegen/box'
    cooperative__send = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/request.txt'
    cooperative__take = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/answer.txt'
    cooperative__paths = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/paths.json'
    cooperative__types = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/types.json'
    cooperative__compared_types = '/home/tihonovcore/diploma/kotlin/idea/tests/org/jetbrains/kotlin/diploma/out/compareTypes.txt'

    cd = 'cd /home/tihonovcore/diploma/kotlin; '
    gradle_extract_paths = cd + './gradlew :idea:test --tests "org.jetbrains.kotlin.idea.caches.resolve.ExtractPaths.testTTT" -q'
    gradle_on_predict = cd + './gradlew :idea:test --tests "org.jetbrains.kotlin.idea.caches.resolve.OnPredict.testTTT" -q'

    request = '/home/tihonovcore/diploma/kotlin/compiler/cli/src/org/jetbrains/kotlin/diploma/actions/request.json'
    _random_kt_file = '/home/tihonovcore/diploma/kotlin/idea/idea-core/src/org/jetbrains/kotlin/idea/core/util/PhysicalFileSystemUtils.kt'
    bash_compiler = '/bin/bash /home/tihonovcore/diploma/kotlin/dist/kotlinc/bin/kotlinc-jvm ' + _random_kt_file

    use_leak_cheat: bool = True
    if use_leak_cheat:
        kotlin_test_directory = join(root_path, 'tests.txt')
