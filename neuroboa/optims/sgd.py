from .optim import Optimizer

import numpy as np

class SGD(Optimizer):
    def __init__(self, lr = 0.01):
        self.lr = lr

    def step(self, wt, grad):
        wt -= self.lr * grad

class SGDMomentum(SGD):
    def __init__(self, lr = 0.01, momentum = 0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.vt = []

    def step(self, wt, grad):
        pass