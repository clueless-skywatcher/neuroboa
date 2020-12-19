from .func import BaseFunction

import numpy as np

class ReLU(BaseFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def gradient(self, grad):
        return np.where(grad >= 0, 1, 0)

class ParamReLU(BaseFunction):
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.minimum(0, x) * self.alpha + np.maximum(0, x)

    def gradient(self, grad):
        return np.where(grad >= 0, 1, self.alpha)

    def __str__(self):
        return f"{self.__class__.__name__}({self.alpha})"

class LeakyReLU(ParamReLU):
    def __init__(self):
        super().__init__(alpha = 0.01)