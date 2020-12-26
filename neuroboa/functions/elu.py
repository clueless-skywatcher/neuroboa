from .func import BaseFunction

import numpy as np

class ELU(BaseFunction):
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, grad):
        return np.where(grad >= 0, 1, self.alpha * np.exp(grad))
