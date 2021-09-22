import subprocess
from os import walk
from os.path import join
from random import shuffle

from configuration import Configuration

if __name__ == '__main__':
    assert Configuration.use_leak_cheat

    file_paths = []

    for root, _, files in walk(Configuration.kotlin_test_directory):
        for file in files:
            file_path = join(root, file)
            file_paths.append(file_path)

    shuffle(file_paths)

    batch = 1
    for from_index in range(0, len(file_paths), batch):
        paths = file_paths[from_index:from_index + batch]
        print(paths)

        with open(Configuration.kotlin_test_directory, 'w') as tests:
            tests.write('\n'.join(paths))

        result = subprocess.run('python3 /home/tihonovcore/diploma/model/active_fit.py', capture_output=True, shell=True)
        print(result.stdout.decode('utf-8'))
        print(result.stderr.decode('utf-8'))

