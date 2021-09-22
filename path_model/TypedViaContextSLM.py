from path_model.abstract_slm import TypedSLM


class TypedViaContextSLM(TypedSLM):
    """Implementation of SLM with type injection via
    context container
    """

    def __init__(self, **kwargs):
        super().__init__(name='structural_language_model__context', **kwargs)

    def call(self, inputs, training=None, mask=None):
        raise Exception('not yet implemented')
