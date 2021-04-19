import tensorflow as tf

from configuration import Configuration
from numpy import argmax


def evaluate_statistics(evaluate_begin, evaluate_end, composed, target_indices, targets, slm, index2word):
    actual = []
    real = []

    ok = 0
    batch_size = Configuration.test_batch_size
    for begin in range(evaluate_begin, evaluate_end, batch_size):
        if begin % 100 == 0:
            print('begin: %d' % begin)

        composed_batch = tf.ragged.constant(composed[begin:begin + batch_size])
        indices_batch = tf.constant(target_indices[begin:begin + batch_size])

        result = slm.call((composed_batch, indices_batch))

        actual_batch = tf.argmax(result, axis=1).numpy()
        real_batch = tf.argmax(targets[begin:begin + batch_size], axis=1).numpy()

        actual.extend(actual_batch)
        real.extend(real_batch)

    for (a, r) in list(zip(actual, real)):
        if (a == r):
            ok += 1

    print('test accuracy: %f' % (ok / (evaluate_end - evaluate_begin)))

    real_stat = {}
    for target in targets[evaluate_begin:evaluate_end]:
        t = argmax(target)
        real_stat.setdefault(t, 0)
        real_stat[t] += 1

    actual_tp_stat = {}
    actual_fp_stat = {}
    for (r, a) in zip(real, actual):
        if r == a:
            actual_tp_stat.setdefault(r, 0)
            actual_tp_stat[r] += 1
        else:
            actual_fp_stat.setdefault(r, {})
            actual_fp_stat[r].setdefault(a, 0)
            actual_fp_stat[r][a] += 1

    print("TRUE POSITIVE STAT")
    for (real_index, real_count) in real_stat.items():
        actual_tp_stat.setdefault(real_index, 0)

        actual_count = actual_tp_stat[real_index]

        print('%d / %d\t= %f \t %s' % (actual_count, real_count, actual_count / real_count, index2word[real_index]))

    print("\n\n\nFALSE POSITIVE STAT")
    for (real_index, real_count) in real_stat.items():
        actual_fp_stat.setdefault(real_index, {})
        stat = actual_fp_stat[real_index]

        if len(stat) == 0:
            continue

        for index, freq in sorted(stat.items(), key=lambda s: s[1], reverse=True)[:3]:
            print('%d \t %s --> %s' % (freq, index2word[real_index], index2word[index]))
