class PreparedData:
    def __init__(
            self,
            composed,
            left_brothers,
            index_among_brothers
    ):
        self.composed = composed
        self.left_brothers = left_brothers
        self.index_among_brothers = index_among_brothers

    def updated(self, new_composed, new_left_brothers):
        return PreparedData(
            new_composed,
            new_left_brothers,
            self.index_among_brothers
        )
