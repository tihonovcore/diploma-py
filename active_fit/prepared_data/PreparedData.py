from tensorflow import RaggedTensor
from tensorflow.python.framework.ops import EagerTensor
from typing import List


class PreparedData:
    def __init__(
            self,
            composed: RaggedTensor,
            left_brothers: RaggedTensor,
            index_among_brothers: EagerTensor
    ):
        self.composed: RaggedTensor = composed
        self.left_brothers: RaggedTensor = left_brothers
        self.index_among_brothers: EagerTensor = index_among_brothers

    def updated(self, new_composed: RaggedTensor, new_left_brothers: RaggedTensor):
        return PreparedData(
            new_composed,
            new_left_brothers,
            self.index_among_brothers
        )
