from .layer import Layer

import numpy as np

import math
import copy

class Dense(Layer):
    def __init__(self, neurons, input_shape = None):
        self.neurons = neurons
        self.input_shape = input_shape
        self.wts = {}

    def _precompute(self):
        limit = 1 / np.sqrt(self.input_shape[0])
        self.wts = {
            "W" : np.random.uniform(-limit, limit, (self.input_shape[0], self.neurons)),
            "b" : np.zeros((1, self.neurons))
        }

    def forward(self, x):
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