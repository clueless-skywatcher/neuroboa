from neuroboa.layers.layer import Layer

import numpy as np

class Padder2D(Layer):
    """
    Implements a Padding layer to pad 2 dimensional values during 
    data forwarding

    Params:
    ---------------------------------------
    val: number
        The value that is to be padded

    padding: integer or a tuple of integers/tuples
        If integer, pad symmetrically on all sides around the input

        If in the form ((up,down), (left, right)), pad "left" values on the
        left, "right" values on the right and so on.

        If in the form (up_down, (left, right)), pad "left" values on the left,
        "right" values on the right, and "up_down" values symmetrically on both
        up and down sides.

        If in the form ((up, down), left_right), pad "up" values up,
        "down" values down, and "left_right" values symmetrically on both
        left and right sides.
    """
    def __init__(self, val, padding = 1):
        self.padding = padding
        self.val = val

        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))        
        elif isinstance(padding[0], tuple) and isinstance(padding[1], int):
            self.padding = ((padding[0][0], padding[0][1]), (padding[1], padding[1]))        
        elif isinstance(padding[1], tuple) and isinstance(padding[0], int):
            self.padding = ((padding[0], padding[0]), (padding[1][0], padding[1][1]))

    def forward(self, X):
        if len(X.shape) != 4:
            raise Exception("Input must be of 4 dimensions: (batch_size, channels, height, width)")
        padded = np.pad(X, pad_width = ((0, 0), (0, 0), (self.padding[0][0], self.padding[0][1]), (self.padding[1][0], self.padding[1][1])), mode = "constant", constant_values = self.val)
        return padded

    def backward(self, grad):
        pass

    def output_shape(self):
        pass

class ZeroPadder2D(Padder2D):
    def __init__(self, padding = 1):
        super(ZeroPadder2D, self).__init__(val = 0.0, padding = padding)

if __name__ == "__main__":
    padder = ZeroPadder2D(padding = ((0, 1), (0, 1)))
    arr = np.array([[[
        [1, 2, 3],
        [3, 4, 5],
        [5, 6, 7]
    ]]])

    print(padder.forward(arr))