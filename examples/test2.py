from neuroboa.layers import Conv1D

import numpy as np

conv = Conv1D(filter_ = np.array([1, 1, 1]), stride=2, padding="same")
print(conv._conv_1dim(np.array([1, 2, 3, 4, 5, 6, 7])))