import tensorflow as tf

from actions.find_possible_children import parent_id_to_children_ids
from actions.process_dataset import ProcessedDataset
from configuration import Configuration
from numpy import argmax


def evaluate_statistics(evaluate_begin, evaluate_end, processed_dataset: ProcessedDataset, slm):
    real = []

    actual = []
    actual_top_5 = []

    gram_acc_1 = []
    gram_acc_5 = []

    grammar_first = []

    batch_size = Configuration.test_batch_size
    for begin in range(evaluate_begin, evaluate_end, batch_size):
        if begin % 100 == 0:
            print('begin: %d' % begin)

        composed_batch = tf.ragged.constant(processed_dataset.composed[begin:begin + batch_size])
        left_brothers_batch = tf.ragged.constant(processed_dataset.left_brothers[begin:begin + batch_size])
        indices_batch = tf.constant(processed_dataset.target_indices[begin:begin + batch_size])
        leaf_types = processed_dataset.types_for_leaf_paths[begin:begin + batch_size]
        root_types = processed_dataset.types_for_root_path[begin:begin + batch_size]
        type_container_id_batch = processed_dataset.type_container_id[begin:begin + batch_size]

        result, _ = slm.call((composed_batch, indices_batch, type_container_id_batch, leaf_types, root_types, processed_dataset.type_container_embeddings))

        empty_children_ids_count = 0
        for (res, cmp, left_brothers) in zip(result, composed_batch, left_brothers_batch):
            parent_id = cmp[-1][-1]
            children_ids = parent_id_to_children_ids(parent_id, left_brothers)

            _, predicted = tf.nn.top_k(tf.gather(res, children_ids), k=1)
            gram_acc_1.append(children_ids[predicted.numpy()[0]])

            _, predicted = tf.nn.top_k(tf.gather(res, children_ids), k=min(5, len(children_ids)))
            gram_acc_5.append(list(map(lambda pred: children_ids[pred], predicted.numpy())))

            _, top_k_indices = tf.nn.top_k(res, len(children_ids))
            intersection = set(top_k_indices.numpy()) & set(children_ids)

            if len(children_ids) != 0:
                grammar_first.append(len(intersection) / len(children_ids))
            else:
                empty_children_ids_count += 1

        _, top_5_indices = tf.nn.top_k(result, 5)
        actual_top_5.extend(top_5_indices)

        actual_batch = tf.argmax(result, axis=1).numpy()
        real_batch = tf.argmax(processed_dataset.targets[begin:begin + batch_size], axis=1).numpy()

        actual.extend(actual_batch)
        real.extend(real_batch)

    ok = 0
    for (a, r) in list(zip(actual, real)):
        if a == r:
            ok += 1

    print('test accuracy@1: %f' % (ok / (evaluate_end - evaluate_begin)))

    ok = 0
    for (a, r) in list(zip(actual_top_5, real)):
        if r in a:
            ok += 1

    print('test accuracy@5: %f' % (ok / (evaluate_end - evaluate_begin)))

    ok = 0
    for (g, r) in list(zip(gram_acc_1, real)):
        if r == g:
            ok += 1

    print('test naive_grammar_accuracy@1:   %f' % (ok / (evaluate_end - evaluate_begin)))

    ok = 0
    for (g, r) in list(zip(gram_acc_5, real)):
        if r in g:
            ok += 1

    print('test naive_grammar_accuracy@<=5: %f' % (ok / (evaluate_end - evaluate_begin)))

    print('test grammar_first: %f' % (sum(grammar_first) / (evaluate_end - evaluate_begin)))
    print('!!! empty_children_ids_count: %d' % empty_children_ids_count)

    real_stat = {}
    for target in processed_dataset.targets[evaluate_begin:evaluate_end]:
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

        print('%d / %d\t= %f \t %s' % (actual_count, real_count, actual_count / real_count, processed_dataset.integer2string[real_index]))

    print("\n\n\nFALSE POSITIVE STAT")
    for (real_index, real_count) in real_stat.items():
        actual_fp_stat.setdefault(real_index, {})
        stat = actual_fp_stat[real_index]

        if len(stat) == 0:
            continue

        for index, freq in sorted(stat.items(), key=lambda s: s[1], reverse=True)[:3]:
            print('%d \t %s --> %s' % (freq, processed_dataset.integer2string[real_index], processed_dataset.integer2string[index]))
