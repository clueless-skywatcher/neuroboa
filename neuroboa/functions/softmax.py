from .func import BaseFunction

import numpy as np

class Softmax(BaseFunction):
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis = -1, keepdims = True))
        return exps / np.sum(exps, axis = -1, keepdims = True)

    def gradient(self, grad):
        g = self.forward(grad)
        return g * (1 - g)