from abc import ABC
from tensorflow import keras


class SLM(keras.Model, ABC):
    def __init__(self, name, **kwargs):
        super(SLM, self).__init__(name=name, **kwargs)


class TypedSLM(SLM, ABC):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
