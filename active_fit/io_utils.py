import re
from os import walk
from os.path import join
from typing import List

from configuration import Configuration


def get_paths_to_snippets() -> List[str]:
    file_paths: List[str] = []

    if Configuration.use_leak_cheat:
        with open(Configuration.kotlin_test_directory, 'r') as tests:
            for file_path in tests.read().split("\n"):
                if must_be_skipped(file_path):
                    continue

                file_paths.append(file_path)
    else:
        for root, _, files in walk(Configuration.kotlin_test_directory):
            for file in files:
                file_path = join(root, file)
                if must_be_skipped(file_path):
                    continue

                file_paths.append(file_path)

    return file_paths


def must_be_skipped(path: str) -> bool:
    if path[-2:] != 'kt':
        return True

    if path.endswith('kt30402.kt') or path.endswith('crossTypeEquals.kt') or path.endswith('jsNative.kt'):
        return True

    with open(path) as source:
        text = source.read()

        if re.search('//\\s*?FILE:', text) is not None:
            return True
        if re.search('//\\s*?WITH_RUNTIME', text) is not None:
            return True
        if re.search('//\\s*?FILE: .*?\\.java', text) is not None:
            return True
