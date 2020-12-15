from .func import BaseFunction

import numpy as np

class ReLU(BaseFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def gradient(self, grad):
        grad[grad <= 0] = 0
        grad[grad > 0] = 1
        return grad