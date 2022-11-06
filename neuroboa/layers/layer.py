from neuroboa.initializers import *


class Layer():
    _INITIALIZERS_DICT = {
        "uniform" : UniformInitializer,
        "glorot_normal" : GlorotNormalInitializer,
        "glorot_uniform" : GlorotUniformInitializer,
        "he_normal" : HeNormalInitializer,
        "he_uniform" : HeUniformInitializer,
        "zero": ZeroInitializer
    }
    def __init__(self, *args, **kwargs):
        self.optimizer = None
        self.wts = {}
        self.input_shape = None
        pass

    def forward(self, x, training = True):
        raise NotImplementedError()

    def backward(self, grad):
        raise NotImplementedError()

    def _set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _overview(self):
        raise NotImplementedError()