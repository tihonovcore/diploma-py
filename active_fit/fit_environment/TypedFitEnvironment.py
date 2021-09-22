from active_fit.fit_environment.FitEnvironment import FitEnvironment
from active_fit.loss.TreeGenerationLoss import TreeGenerationLoss
from active_fit.loss.TypedTreeGenerationLoss import TypedTreeGenerationLoss
from path_model.slm import SLM
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from type_model.implementation import TE
from typing import List


class TypedFitEnvironment(FitEnvironment):
    def __init__(self, file_paths: List[str], type_embeddings: TE, slm: SLM, optimizer: OptimizerV2):
        super().__init__(file_paths, slm, optimizer)
        self.type_embeddings: TE = type_embeddings

    def values(self) -> (List[str], TE, SLM, OptimizerV2):
        return self.file_paths, self.type_embeddings, self.slm, self.optimizer

    @staticmethod
    def create_loss() -> TreeGenerationLoss:
        return TypedTreeGenerationLoss()
