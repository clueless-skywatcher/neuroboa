from .func import BaseFunction
from .sigmoid import Sigmoid

class Swish(BaseFunction):
    def forward(self, x):
        s = Sigmoid()
        return x * s(x)

    def gradient(self, grad):
        s = Sigmoid()
        return s.gradient(grad) * grad + s(grad)