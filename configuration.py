from os.path import abspath, curdir, join


class Configuration:
    root_path = abspath(curdir)
    if root_path.endswith('content'):
        root_path = join(root_path, 'model')

    integer2string_json = join(root_path, 'dataset', 'integer2string.json')
    string2integer_json = join(root_path, 'dataset', 'string2integer.json')
    train_dataset_json = join(root_path, 'dataset', 'dataset.json')

    saved_model = join(root_path, 'saved_model', 'model')
