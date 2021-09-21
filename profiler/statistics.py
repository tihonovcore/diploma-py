import json
from os import walk
from os.path import join


def show_statistics():
    file_paths = []
    for root, _, files in walk('/home/tihonovcore/diploma/model/profiler/out'):
        for file in files:
            file_path = join(root, file)
            file_paths.append(file_path)

    file_paths = sorted(file_paths)
    files_stats = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            text = json.loads(file.read())
            files_stats.append(text)

    ten_heaviest = find_n_heaviest(10, files_stats)
    for file_stat in files_stats:
        print('@@@ %s @@@@@@@@@@@@' % file_stat['msg'])
        for t in ten_heaviest:
            count = file_stat['map_types'][t]
            weight = file_stat['map_weights'][t]
            print('%s ||| count: %d ||| size: %d' % (t, count, weight))
        print()

    print('spend memory %d' % sum(map(lambda p: p[1], files_stats[-1]['map_weights'].items())))


def find_n_heaviest(n, files_stats):
    items = files_stats[-1]['map_weights'].items()
    items = sorted(items, key=lambda item: item[1], reverse=True)
    items = list(map(lambda item: item[0], items))

    n = min(n, len(items))
    return items[:n]
