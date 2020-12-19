from .loss import LossFunction

import numpy as np

class BinaryCrossEntropy(LossFunction):
    def __init__(self):
        pass

    def loss(self, x, y):
        y = np.clip(y, 1e-15, 1 - 1e-15)
        return -np.mean(x * np.log(y) + (1 - x) * np.log(1 - y), axis = 1)

    def gradient(self, x, y):
        y = np.clip(y, 1e-15, 1 - 1e-15)
        return -(x / y) + (1 - x) / (1 - y)

    