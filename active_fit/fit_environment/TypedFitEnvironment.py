from active_fit.fit_environment.FitEnvironment import FitEnvironment
from active_fit.loss.TreeGenerationLoss import TreeGenerationLoss
from active_fit.loss.TypedTreeGenerationLoss import TypedTreeGenerationLoss
from path_model.abstract_slm import TypedSLM
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from type_model.implementation import TE
from typing import List


class TypedFitEnvironment(FitEnvironment):
    def __init__(self, file_paths: List[str], type_embeddings: TE, slm: TypedSLM, optimizer: OptimizerV2):
        super().__init__(file_paths, slm, optimizer)
        self.type_embeddings: TE = type_embeddings

    @staticmethod
    def create_loss() -> TreeGenerationLoss:
        return TypedTreeGenerationLoss()
