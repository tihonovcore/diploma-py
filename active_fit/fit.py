from typing import List

from configuration import Configuration
from active_fit.initialize import FitEnvironment
from active_fit.communication import RequestType, call_kotlin_compiler, ResponseStatus

import tensorflow as tf

from active_fit.prepare_data import prepare_data, PreparedData
from active_fit.loss import TreeGenerationLoss
from active_fit.predict import predict


def fit(fit_environment: FitEnvironment):
    file_paths, type_embeddings, slm, optimizer = fit_environment.values()

    for file_number, file_path in enumerate(file_paths):
        if file_number % 5 == 0:
            slm.save_weights(Configuration.saved_model)

        print(file_path)

        with tf.GradientTape() as tape:
            loss: TreeGenerationLoss = fit_environment.create_loss()

            status: ResponseStatus = call_kotlin_compiler(RequestType.EXTRACT_PATHS, file_path)
            while status is ResponseStatus.PATH:
                request: List[str] = []
                prepared_data: PreparedData = prepare_data(type_embeddings)
                predict(prepared_data, slm, loss, request)

                print('##########')
                status = call_kotlin_compiler(RequestType.ON_PREDICT, '\\n'.join(request))

            loss.eval_full_loss(status)
            loss.print_loss()

        if status is ResponseStatus.SUCC:
            grads = tape.gradient(loss.get_full_loss(), slm.trainable_weights)
            optimizer.apply_gradients(zip(grads, slm.trainable_weights))

            print("last loss = %.4f" % loss.get_full_loss())
        elif status is ResponseStatus.FAIL:
            grads = tape.gradient(loss.get_full_loss(), slm.trainable_weights)
            optimizer.apply_gradients(zip(grads, slm.trainable_weights))

            print("last loss = %.4f" % loss.get_full_loss())

        slm.save_weights(Configuration.saved_model)
