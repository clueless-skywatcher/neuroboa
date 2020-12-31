from .layer import Layer

class Activation(Layer):
    def __init__(self, func):        
        self.func = func
        self.input_shape = None

    def forward(self, x, training = True):
        self.input = x
        return self.func.forward(x)

    def backward(self, grad):
        return grad * self.func.gradient(self.input)

    def output_shape(self):
        return self.input_shape

    def _overview(self):
        return [self.__class__.__name__, str(self.func)]
