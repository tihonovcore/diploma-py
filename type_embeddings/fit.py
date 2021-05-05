import json
from os import walk
from os.path import join

import tensorflow as tf

from type_embeddings.question import Question

# todo:
#  Model has a problem:
#  let there are three nodes: a, b, c
#  and three orientated edges: (a, b) (a, c) (b, c)
#  on backward step node `c` gets loss twice.
#  For this reason loss function grow while learning.
if __name__ == '__main__':
    file_names = []
    for (dirpath, dirnames, filenames) in walk('/home/tihonovcore/diploma/kotlin/compiler/tests-common-new/tests/org/jetbrains/kotlin/test/backend/handlers/dataset'):
        for name in filenames:
            file_names.append(join(dirpath, name))

    model = Question(actions_per_question=50)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')

    # step = 1
    #
    # def my_loss(y_true, y_pred):
    #     return - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)) / step / step

    for epoch in range(5):
        print('start epoch %d' % epoch)
        for file_number, name in enumerate(file_names):
            # step += 1

            with open(name, 'r') as file:
                inputs = json.load(file)

            with tf.GradientTape() as tape:
                actual, real = model(inputs)

                metric.update_state(y_true=real, y_pred=actual)

                avg_ls = 0
                for (a, r) in zip(actual, real):
                    ls = loss(y_true=tf.constant(r, shape=(1, 1)), y_pred=a)
                    avg_ls += ls

            grads = tape.gradient(ls, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # if (file_number + 1) % (len(file_names) / 10) == 0:
            percent = (file_number + 1) / (len(file_names) / 100)
            print("%.4f%% metric    = %.4f" % (percent, metric.result()))
            print("%.4f%% avg  ls   = %.4f" % (percent, avg_ls / len(actual)))
