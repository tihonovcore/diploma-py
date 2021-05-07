import json
from os import walk
from os.path import join

import tensorflow as tf

from configuration import Configuration
from type_embeddings.question import Question

if __name__ == '__main__':
    file_names = []
    for (dirpath, dirnames, filenames) in walk('/home/tihonovcore/diploma/kotlin/compiler/tests-common-new/tests/org/jetbrains/kotlin/test/backend/handlers/dataset'):
        for name in filenames:
            file_names.append(join(dirpath, name))

    model = Question()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    for epoch in range(Configuration.type_embedding_model_epochs_count):
        print('start epoch %d' % epoch)
        for file_number, name in enumerate(file_names):

            with open(name, 'r') as file:
                inputs = json.load(file)

            with tf.GradientTape() as tape:
                actual, real = model(inputs)

                metric.update_state(y_true=[real], y_pred=[actual])
                ls = loss(y_true=tf.constant(real, shape=(1, 1)), y_pred=actual)

            grads = tape.gradient(ls, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            percent = (file_number + 1) / (len(file_names) / 100)
            print("%.4f%% metric    = %.4f" % (percent, metric.result()))

        print(list(map(lambda a: a[0] / a[1] if a[1] != 0 else -1, zip(model.ok, model.cnt))))
        print(model.cnt)
