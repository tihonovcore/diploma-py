from ModelMode import ModelMode
from active_fit.fit_environment.FitEnvironment import FitEnvironment
from active_fit.fit_environment.TypedFitEnvironment import TypedFitEnvironment
from configuration import Configuration
from path_model.slm import SLM
from active_fit.io_utils import get_paths_to_snippets
from type_model.question_model import QuestionModel

import tensorflow as tf


def initialize() -> FitEnvironment:
    file_paths = get_paths_to_snippets()

    slm = SLM(batch_size=20)
    slm.load_weights(Configuration.saved_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if Configuration.model_mode is ModelMode.UNTYPED:
        return FitEnvironment(file_paths, slm, optimizer)

    question_model: QuestionModel = QuestionModel(mode=Configuration.recurrent_mode)
    question_model.trainable = False
    question_model.load_weights(Configuration.saved_type_model)
    type_embeddings = question_model.type_embeddings

    return TypedFitEnvironment(file_paths, type_embeddings, slm, optimizer)
