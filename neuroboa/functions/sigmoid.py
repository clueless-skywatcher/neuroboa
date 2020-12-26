from .func import BaseFunction

import numpy as np

class Sigmoid(BaseFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, grad):
        s = 1 / (1 + np.exp(-grad))
        return s * (1 - s)