from .layer import Layer

import numpy as np

class Dropout(Layer):
    def __init__(self, keep = 0.8):
        self.keep = keep


    def forward(self, input_, training = True):
        self.input = input_
        self.input_shape = input_.shape
        if not training:
            return self.input * self.keep
        
        self._mask = np.random.binomial(1, self.keep, size = self.input.shape)
        return self.input * self._mask

    def backward(self, grad):
        return grad * self._mask

    def output_shape(self):
        return self.input_shape

    def _overview(self):
        return [self.__class__.__name__, self.keep]