from .func import BaseFunction

class LinearFunction(BaseFunction)
    def __init__(self, a = 1.0):
        self.a = a
    
    def forward(self, x):
        return self.a * x

    def gradient(self, grad):
        return self.a