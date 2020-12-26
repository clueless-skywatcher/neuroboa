from .loss import LossFunction

import numpy as np

class MSE(LossFunction):
    def loss(self, x, y):
        return np.mean((x - y) ** 2)

    def gradient(self, x, y):
        return - 2 * (x - y)