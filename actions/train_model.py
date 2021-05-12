from actions.find_possible_children import get_weights_batch
from actions.process_dataset import ProcessedDataset
from configuration import Configuration
from implementation.slm import SLM

import tensorflow as tf


def loss(real, actual, weights):
    result = []
    for (a, w) in zip(actual, weights):
        result.append(tf.reduce_sum(-tf.math.log(1 - tf.gather(a, w))))
    return tf.convert_to_tensor(result)


def train_model(processed_dataset: ProcessedDataset, slm=SLM(batch_size=20)):
    dataset_size = Configuration.train_dataset_size

    all_features = tf.ragged.constant(processed_dataset.composed[:dataset_size], dtype='float32')
    all_brothers = tf.ragged.constant(processed_dataset.left_brothers[:dataset_size])
    all_target_indices = tf.constant(processed_dataset.target_indices[:dataset_size])
    all_targets = tf.constant(processed_dataset.targets[:dataset_size])
    all_leaf_types = processed_dataset.types_for_leaf_paths[:dataset_size]
    all_root_types = processed_dataset.types_for_root_path[:dataset_size]
    all_type_container_id = processed_dataset.type_container_id[:dataset_size]

    batch_size = Configuration.train_batch_size
    train_features = [all_features[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_brothers = [all_brothers[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_target_indices = [all_target_indices[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_targets = [all_targets[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_leaf_types = [all_leaf_types[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_root_types = [all_root_types[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_type_container_id = [all_type_container_id[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_dataset = list(zip(train_features, train_brothers, train_target_indices, train_type_container_id, train_leaf_types, train_root_types, train_targets))

    # slm = SLM(batch_size=20)  # todo: wtf is batch_sze?

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    for epoch in range(Configuration.epochs_count):
        print("Start of epoch %d" % epoch)

        for batch_number, (x_batch, brothers_batch, indices_batch, type_container_id_batch, leaf_types, root_types, y_batch) in enumerate(train_dataset):
            impossible_children_batch = get_weights_batch(x_batch, brothers_batch)

            with tf.GradientTape() as tape:
                reconstructed = slm((x_batch, indices_batch, type_container_id_batch, leaf_types, root_types, processed_dataset.type_container_embeddings))

                metric.update_state(y_pred=reconstructed, y_true=tf.argmax(y_batch, axis=1))
                ls = loss(y_batch, reconstructed, impossible_children_batch)

            grads = tape.gradient(ls, slm.trainable_weights)
            optimizer.apply_gradients(zip(grads, slm.trainable_weights))

            if ((batch_number + 1) * batch_size) % (dataset_size / 10) == 0:
                percent = ((batch_number + 1) * batch_size) / (dataset_size / 100)
                print("%d%% METRIC        = %.4f" % (percent, metric.result()))
                print("%d%% LAST AVG LOSS = %.4f" % (percent, tf.reduce_sum(ls) / batch_size))

    return slm
