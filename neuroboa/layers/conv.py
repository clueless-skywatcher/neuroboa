from .layer import Layer

import numpy as np

class Conv1D(Layer):
    def __init__(self, filter_ = [1, 1, 1], stride = 1, padding = "same"):
        self.filter = np.array(filter_)
        self.stride = stride
        self.padding = padding

    def _pad_1dim(self, input_, padding = 1):
        if isinstance(padding, int):
            padding = (padding, padding)
        zero = np.array([0])
        return np.concatenate([
            np.repeat(zero, padding[0]), 
            input_, 
            np.repeat(zero, padding[1])
        ])

    def _conv_1dim(self, input_):
        if self.filter.shape[0] % 2 == 0:
            raise Exception("Filter size must be odd")
        if self.padding == "same":
            pad_size = (input_.shape[0] * (self.stride - 1) + self.filter.shape[0] - 1) // 2
            input_ = self._pad_1dim(input_, pad_size)        
        
        output = np.array([])
        for i in range(0, len(input_) - len(self.filter) + 1, self.stride):
            output = np.append(output, np.dot(input_[i : i + len(self.filter)], self.filter))

        return output

    def forward(self, x):
        self.input = x
        self.input_shape = x.shape

        return self._conv_1dim(x)
        

