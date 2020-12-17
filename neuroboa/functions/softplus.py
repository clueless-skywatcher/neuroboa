from .func import BaseFunction

import numpy as np

class Softplus(BaseFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, grad):
        return 1 / (1 + np.exp(-grad))