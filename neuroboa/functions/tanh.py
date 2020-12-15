from .func import BaseFunction

import numpy as np

class Tanh(BaseFunction):
    def forward(self, x):
        return np.tanh(x)

    def gradient(self, grad):
        return 1 - (np.tanh(grad) ** 2)