from active_fit.prepared_data.PreparedData import PreparedData


class TypedPreparedData(PreparedData):
    def __init__(
            self,
            composed,
            left_brothers,
            type_container_id,
            type_container_embeddings,
            leaf_types,
            root_types,
            index_among_brothers
    ):
        super().__init__(composed, left_brothers, index_among_brothers)

        self.type_container_id = type_container_id
        self.type_container_embeddings = type_container_embeddings
        self.leaf_types = leaf_types
        self.root_types = root_types

    def updated(self, new_composed, new_left_brothers):
        return TypedPreparedData(
            new_composed,
            new_left_brothers,
            self.type_container_id,
            self.type_container_embeddings,
            self.leaf_types,
            self.root_types,
            self.index_among_brothers
        )
