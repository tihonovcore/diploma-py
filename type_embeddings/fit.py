import json
import random
from os import walk
from os.path import join

import tensorflow as tf

from configuration import Configuration
from type_embeddings.generate_questions import generate_questions
from type_embeddings.process_questions import process_questions
from type_embeddings.question_model import QuestionModel
from type_embeddings.question_statistics import print_question_statistics

if __name__ == '__main__':
    file_names = []
    for (dirpath, dirnames, filenames) in walk(Configuration.types_dataset):
        for name in filenames:
            file_names.append(join(dirpath, name))

    model = QuestionModel(mode='lstm')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    samples_cnt = len(file_names)
    test_size = samples_cnt // 10
    train_size = samples_cnt - test_size

    random.shuffle(file_names)
    train_names = file_names[:train_size]
    test_names = file_names[train_size:]

    print('start question generation')
    questions = []
    for file_number, name in enumerate(file_names):
        with open(name, 'r') as file:
            inputs = json.load(file)

        questions.append(generate_questions(inputs, Configuration.questions_per_file_train))
    print('questions have generated')

    process_questions(questions)
    print_question_statistics(questions)

    train_questions = questions[:train_size]
    test_questions = questions[train_size:]

    for epoch in range(Configuration.type_embedding_model_epochs_count):
        print('start epoch %d' % epoch)
        for file_number, (name, questions_for_this_file) in enumerate(zip(train_names, train_questions)):

            with open(name, 'r') as file:
                inputs = json.load(file)

            for single_question in questions_for_this_file:
                with tf.GradientTape() as tape:
                    try:
                        actual, real = model([inputs, single_question])
                    except BaseException as err:
                        print(err)
                        continue

                    metric.update_state(y_true=[real], y_pred=[actual])
                    ls = loss(y_true=tf.constant(real, shape=(1, 1)), y_pred=actual)

                grads = tape.gradient(ls, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            percent = (file_number + 1) / (train_size / 100)
            print("%.4f%% metric    = %.4f" % (percent, metric.result()))

            if file_number % 50 == 0:
                model.save_weights(Configuration.saved_type_model)

                print(list(map(lambda a: a[0] / a[1] if a[1] != 0 else -1, zip(model.ok, model.cnt))))
                print(model.cnt)

        print(list(map(lambda a: a[0] / a[1] if a[1] != 0 else -1, zip(model.ok, model.cnt))))
        print(model.cnt)

    model.save_weights(Configuration.saved_type_model)

    model = QuestionModel()
    model.load_weights(Configuration.saved_type_model)

    model.ok = [0 for _ in range(model.question_count)]
    model.cnt = [0 for _ in range(model.question_count)]

    test_metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    for file_number, (name, questions_for_this_file) in enumerate(zip(test_names, test_questions)):
        with open(name, 'r') as file:
            inputs = json.load(file)

        for q in questions_for_this_file:
            try:
                actual, real = model([inputs, q])
            except BaseException as err:
                print(err)
                continue

            test_metric.update_state(y_true=[real], y_pred=[actual])

        percent = (file_number + 1) / (test_size / 100)
        print("%.4f%% test_metric = %.4f" % (percent, test_metric.result()))

    print(list(map(lambda a: a[0] / a[1] if a[1] != 0 else -1, zip(model.ok, model.cnt))))
    print(model.cnt)
