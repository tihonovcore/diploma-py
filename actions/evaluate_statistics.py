import tensorflow as tf
from numpy import argmax


def evaluate_statistics(TEST_BEGIN, TEST_END, composed, targets, slm, index2word):
    actual = []
    real = []

    ok = 0
    step = 10
    for b in range(TEST_BEGIN, TEST_END, step):
        if b % 100 == 0:
            print('b: %d' % b)

        result = slm.call(tf.ragged.constant(composed[b:b + step]))

        actual_batch = tf.argmax(result, axis=1).numpy()
        real_batch = tf.argmax(targets[b:b + step], axis=1).numpy()

        actual.extend(actual_batch)
        real.extend(real_batch)

    for (a, r) in list(zip(actual, real)):
        if (a == r):
            ok += 1

    print('test accuracy: %f' % (ok / (TEST_END - TEST_BEGIN)))

    real_stat = {}
    for target in targets[TEST_BEGIN:TEST_END]:
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
