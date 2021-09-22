from typing import List

from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from ModelMode import ModelMode
from active_fit.loss import TreeGenerationLoss, TypedTreeGenerationLoss
from configuration import Configuration
from path_model.slm import SLM
from active_fit.io_utils import get_paths_to_snippets
from type_model.implementation import TE
from type_model.question_model import QuestionModel

import tensorflow as tf


class FitEnvironment:
    def __init__(self, file_paths: List[str], slm: SLM, optimizer: OptimizerV2):
        self.file_paths: List[str] = file_paths
        self.slm: SLM = slm
        self.optimizer: OptimizerV2 = optimizer

    def values(self) -> (List[str], SLM, OptimizerV2):
        return self.file_paths, self.slm, self.optimizer

    @staticmethod
    def create_loss() -> TreeGenerationLoss:
        return TreeGenerationLoss()


class TypedFitEnvironment(FitEnvironment):
    def __init__(self, file_paths: List[str], type_embeddings: TE, slm: SLM, optimizer: OptimizerV2):
        super().__init__(file_paths, slm, optimizer)
        self.type_embeddings: TE = type_embeddings

    def values(self) -> (List[str], TE, SLM, OptimizerV2):
        return self.file_paths, self.type_embeddings, self.slm, self.optimizer

    @staticmethod
    def create_loss() -> TreeGenerationLoss:
        return TypedTreeGenerationLoss()


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
