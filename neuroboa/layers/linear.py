from .layer import Layer
from ..initializers import *

import numpy as np

import math
import copy

class Dense(Layer):
    """
    Implements the Dense (or linear) layer for output calculation.
    Dense means that the layer has some units called neurons which help
    in training the weights of the model.

    The forward function of Dense layer implements the operation
    y = x * W + b

    Optimizers are applied in the backward function itself

    Params:
    ------------------
    neurons: int 
        The number of neurons you want to train the layer

    input_shape: Tuple
        Input dimension (typically the number of features in your dataset).
        This is absolutely needed to be passed if this is the first layer of your
        network. If its not the first layer, the input shape is inferred from the 
        previous layer itself.
    
    initializer: str
        Specifies what initialization technique is to be applied on the weights. Can
        be any of the following:
            - uniform
            - glorot_normal
            - glorot_uniform
            - he_normal
            - he_uniform
    """

    _INITIALIZERS_DICT = {
        "uniform" : UniformInitializer,
        "glorot_normal" : GlorotNormalInitializer,
        "glorot_uniform" : GlorotUniformInitializer,
        "he_normal" : HeNormalInitializer,
        "he_uniform" : HeUniformInitializer
    }

    def __init__(self, neurons, input_shape = None, initializer = "uniform"):
        self.neurons = neurons
        self.input_shape = input_shape
        self.wts = {}
        if initializer not in self._INITIALIZERS_DICT:
            raise Exception(f"Cannot find any initializers by the name: {initializer}")
        else:
            self.initializer = initializer

    def _precompute(self):
        self.wts = {
            "W" : self._INITIALIZERS_DICT[self.initializer](self.input_shape[0], self.neurons).initialize(),
            "b" : np.zeros((1, self.neurons))
        }

    def forward(self, x, training = True):
        if len(self.wts) == 0:
            self._precompute()
        self.input = x
        return x.dot(self.wts["W"]) + self.wts["b"]

    def backward(self, grad):
        w = self.wts["W"]
        
        dw = self.input.T.dot(grad)
        db = np.sum(grad, axis = 0, keepdims = True)

        self.param_optimizers = {
            "W" : copy.copy(self.optimizer),
            "b" : copy.copy(self.optimizer)
        }

        self.param_optimizers["W"].step(self.wts["W"], dw)
        self.param_optimizers["b"].step(self.wts["b"], db)

        grad = grad.dot(w.T)
        return grad
    
    def output_shape(self):
        return (self.neurons, )

    def _overview(self):
        return [self.__class__.__name__, self.neurons, self.input_shape]