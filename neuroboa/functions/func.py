import numpy as np

class BaseFunction():
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError()

    def gradient(self, grad):
        raise NotImplementedError()