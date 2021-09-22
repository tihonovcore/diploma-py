from active_fit.prepared_data.PreparedData import PreparedData
from tensorflow import RaggedTensor
from tensorflow.python.framework.ops import EagerTensor
from typing import List, Dict


class TypedPreparedData(PreparedData):
    def __init__(
            self,
            composed: RaggedTensor,
            left_brothers: RaggedTensor,
            type_container_id: List[int],
            type_container_embeddings: List[EagerTensor],
            leaf_types: List[List[Dict[str, int]]],
            root_types: List[Dict[str, int]],
            index_among_brothers: EagerTensor
    ):
        super().__init__(composed, left_brothers, index_among_brothers)

        self.type_container_id: List[int] = type_container_id
        self.type_container_embeddings: EagerTensor = type_container_embeddings
        self.leaf_types: List[List[Dict[str, int]]] = leaf_types
        self.root_types: List[Dict[str, int]] = root_types

    def updated(self, new_composed: RaggedTensor, new_left_brothers: RaggedTensor):
        return TypedPreparedData(
            new_composed,
            new_left_brothers,
            self.type_container_id,
            self.type_container_embeddings,
            self.leaf_types,
            self.root_types,
            self.index_among_brothers
        )
