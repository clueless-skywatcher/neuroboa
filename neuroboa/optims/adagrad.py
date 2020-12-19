from .optim import Optimizer

import numpy as np

class Adagrad(Optimizer):
    def __init__(self, lr = 0.01, epsilon = 1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.grad_sum = None

    def step(self, weight, grad):
        if self.grad_sum is None:
            self.grad_sum = np.zeros(np.shape(weight))

        self.grad_sum += grad ** 2
        weight -= self.lr * (grad / np.sqrt(self.grad_sum + self.epsilon))