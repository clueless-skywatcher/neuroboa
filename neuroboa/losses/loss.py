class LossFunction():
    def loss(self, x, y, *args, **kwargs):
        raise NotImplementedError

    def gradient(self, x, y, *args, **kwargs):
        raise NotImplementedError