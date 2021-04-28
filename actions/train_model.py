from actions.process_dataset import ProcessedDataset
from configuration import Configuration
from implementation.slm import SLM

import tensorflow as tf


def loss(real, actual, weights):
    diff = tf.subtract(real, actual)
    ls = tf.multiply(weights, diff)
    ls = tf.multiply(ls, diff)
    return tf.reduce_sum(ls, axis=1)


def train_model(processed_dataset: ProcessedDataset, slm=SLM(batch_size=20)):
    dataset_size = Configuration.train_dataset_size

    all_features = tf.ragged.constant(processed_dataset.composed[:dataset_size], dtype='float32')
    all_target_indices = tf.constant(processed_dataset.target_indices[:dataset_size])
    all_targets = tf.constant(processed_dataset.targets[:dataset_size])
    # all_targets = tf.argmax(all_targets, axis=1)
    all_weights = tf.constant(processed_dataset.loss_weights[:dataset_size])

    batch_size = Configuration.train_batch_size
    train_features = [all_features[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_target_indices = [all_target_indices[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_targets = [all_targets[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_weights = [all_weights[i:i + batch_size] for i in range(0, dataset_size, batch_size)]
    train_dataset = list(zip(train_features, train_target_indices, train_targets, train_weights))

    # slm = SLM(batch_size=20)  # todo: wtf is batch_sze?

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    for epoch in range(Configuration.epochs_count):
        print("Start of epoch %d" % epoch)

        for batch_number, (x_batch, indices_batch, y_batch, weights_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = slm((x_batch, indices_batch))

                metric.update_state(y_pred=reconstructed, y_true=tf.argmax(y_batch, axis=1))
                ls = loss(y_batch, reconstructed, weights_batch)

            grads = tape.gradient(ls, slm.trainable_weights)
            optimizer.apply_gradients(zip(grads, slm.trainable_weights))

            if ((batch_number + 1) * batch_size) % (dataset_size / 10) == 0:
                percent = ((batch_number + 1) * batch_size) / (dataset_size / 100)
                print("%d%% METRIC        = %.4f" % (percent, metric.result()))
                print("%d%% LAST AVG LOSS = %.4f" % (percent, tf.reduce_sum(ls) / batch_size))

    return slm
