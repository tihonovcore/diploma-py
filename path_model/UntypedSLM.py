from path_model.abstract_slm import SLM


class UntypedSLM(SLM):
    """Implementation of SLM without type injection
    """

    def __init__(self, **kwargs):
        super().__init__(name='structural_language_model__untyped', **kwargs)

    def call(self, inputs, training=None, mask=None):
        raise Exception('not yet implemented')
