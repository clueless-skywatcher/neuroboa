from .loss import LossFunction
from ..functions import Softmax

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

class SoftmaxCrossEntropy(LossFunction):
    def loss(self, x, y):
        y = np.clip(y, 1e-15, 1 - 1e-15)
        softmax = Softmax()
        return np.sum(-y * np.log(softmax(x)) - (1 - y) * np.log(1 - softmax(x)))

    def gradient(self, x, y):
        softmax = Softmax()
        return softmax(x) - y

    