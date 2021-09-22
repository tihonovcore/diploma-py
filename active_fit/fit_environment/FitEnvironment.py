from active_fit.loss.TreeGenerationLoss import TreeGenerationLoss
from path_model.abstract_slm import SLM
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import List


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
