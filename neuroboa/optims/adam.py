from .optim import Optimizer

import numpy as np

class Adam(Optimizer):
    def __init__(self, 
        lr = 0.001, 
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def step(self, wt, grad):
        if self.m is None:
            self.m = np.zeros(grad.shape)
            self.v = np.zeros(grad.shape)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        wt -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)