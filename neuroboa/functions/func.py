import numpy as np

class BaseFunction():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError()

    def gradient(self, grad):
        raise NotImplementedError()