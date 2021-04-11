from implementation.slm import SLM
from random import shuffle

import tensorflow as tf


def train_model(composed, targets, USE_N_SAMPLES):
    all_features = tf.ragged.constant(composed[:USE_N_SAMPLES], dtype='float32')
    all_targets = tf.constant(targets[:USE_N_SAMPLES])
    all_targets = tf.argmax(all_targets, axis=1)

    BATCH_SIZE = 10
    TRAIN_SIZE = USE_N_SAMPLES
    train_features = [all_features[i:i + BATCH_SIZE] for i in range(0, TRAIN_SIZE, BATCH_SIZE)]
    train_targets = [all_targets[i:i + BATCH_SIZE] for i in range(0, TRAIN_SIZE, BATCH_SIZE)]
    train_dataset = list(zip(train_features, train_targets))

    shuffle(train_dataset)

    slm = SLM(vocab_size=110, embedding_dim=32, batch_size=USE_N_SAMPLES, rnn_units=64, ff_dim=64)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    for epoch in range(2):
        print("Start of epoch %d" % epoch)

        for batch_number, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = slm(x_batch)

                metric.update_state(y_pred=reconstructed, y_true=y_batch)
                ls = loss(y_batch, reconstructed)

            grads = tape.gradient(ls, slm.trainable_weights)
            optimizer.apply_gradients(zip(grads, slm.trainable_weights))

            if batch_number % (TRAIN_SIZE / 100) == 0:
                percent = batch_number / (TRAIN_SIZE / 100)
                print("%d METRIC = %.4f" % (percent, metric.result()))

    return slm
