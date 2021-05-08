import json
from os import walk
from os.path import join

import tensorflow as tf

from configuration import Configuration
from type_embeddings.generate_questions import generate_questions
from type_embeddings.question_model import QuestionModel

if __name__ == '__main__':
    file_names = []
    for (dirpath, dirnames, filenames) in walk('/home/tihonovcore/diploma/kotlin/compiler/tests-common-new/tests/org/jetbrains/kotlin/test/backend/handlers/dataset'):
        for name in filenames:
            file_names.append(join(dirpath, name))

    model = QuestionModel()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print('start question generation')
    questions = []
    for file_number, name in enumerate(file_names):
        with open(name, 'r') as file:
            inputs = json.load(file)

        questions.append(generate_questions(inputs, Configuration.questions_per_file))
    print('questions have generated')

    for epoch in range(Configuration.type_embedding_model_epochs_count):
        print('start epoch %d' % epoch)
        for file_number, (name, questions_for_this_file) in enumerate(zip(file_names, questions)):

            with open(name, 'r') as file:
                inputs = json.load(file)

            for single_question in questions_for_this_file:
                with tf.GradientTape() as tape:
                    actual, real = model([inputs, single_question])

                    metric.update_state(y_true=[real], y_pred=[actual])
                    ls = loss(y_true=tf.constant(real, shape=(1, 1)), y_pred=actual)

                grads = tape.gradient(ls, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            percent = (file_number + 1) / (len(file_names) / 100)
            print("%.4f%% metric    = %.4f" % (percent, metric.result()))

        print(list(map(lambda a: a[0] / a[1] if a[1] != 0 else -1, zip(model.ok, model.cnt))))
        print(model.cnt)
