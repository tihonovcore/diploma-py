import tensorflow as tf

from ModelMode import ModelMode
from active_fit.fit_environment.FitEnvironment import FitEnvironment
from active_fit.fit_environment.TypedFitEnvironment import TypedFitEnvironment
from configuration import Configuration
from active_fit.io_utils import get_paths_to_snippets
from path_model.TypedViaContextSLM import TypedViaContextSLM
from path_model.TypedViaNodesSLM import TypedViaNodesSLM
from path_model.UntypedSLM import UntypedSLM
from type_model.question_model import QuestionModel


def initialize() -> FitEnvironment:
    file_paths = get_paths_to_snippets()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if Configuration.model_mode is ModelMode.UNTYPED:
        slm = UntypedSLM()
        slm.load_weights(Configuration.saved_model)
        return FitEnvironment(file_paths, slm, optimizer)

    question_model: QuestionModel = QuestionModel(mode=Configuration.recurrent_mode)
    question_model.trainable = False
    question_model.load_weights(Configuration.saved_type_model)
    type_embeddings = question_model.type_embeddings

    if Configuration.model_mode is ModelMode.TYPED__INJECTION_VIA_NODES:
        slm = TypedViaNodesSLM()
    elif Configuration.model_mode is ModelMode.TYPED__INJECTION_VIA_CONTEXT:
        slm = TypedViaContextSLM()
    else:
        raise Exception('Unsupported ModelMode: ' + Configuration.model_mode.to_string())
    slm.load_weights(Configuration.saved_model)

    return TypedFitEnvironment(file_paths, type_embeddings, slm, optimizer)
