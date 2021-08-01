class Layer():
    def __init__(self, *args, **kwargs):
        self.optimizer = None
        self.wts = {}
        self.input_shape = None
        pass

    def forward(self, x, training = True):
        raise NotImplementedError()

    def backward(self, grad):
        raise NotImplementedError()

    def _set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _overview(self):
        raise NotImplementedError()