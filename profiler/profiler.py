import gc
import json
import sys


class Profiler:
    step = 0

    @staticmethod
    def log(msg):
        map_types = {}
        map_weights = {}
        for i in gc.get_objects():
            t = str(type(i))

            map_types.setdefault(t, 0)
            map_types[t] += 1

            map_weights.setdefault(t, 0)
            map_weights[t] += sys.getsizeof(type(i))

        output = {'msg': msg, 'map_types': map_types, 'map_weights': map_weights}
        with open('/home/tihonovcore/diploma/model/profiler/out/%s_log.txt' % Profiler.step, 'w') as file:
            file.write(json.dumps(output))

        Profiler.step += 1
