from neuroboa.nn import NN
from neuroboa.layers import Dense, Activation
from neuroboa.functions import ReLU, Tanh, Sigmoid
from neuroboa.optims import SGD, Adam
from neuroboa.losses import BinaryCrossEntropy

import numpy as np

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

outputs = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

nn = NN()
nn.add(Dense(50, input_shape = (2,)))
nn.add(Activation(Tanh()))
nn.add(Dense(2))
nn.add(Activation(Sigmoid()))

nn.fit(inputs, outputs, batch_size = 32, optimizer = Adam(), loss = BinaryCrossEntropy(), epochs = 2000)

pred = nn.predict(np.array([[1, 0], [0, 0]]))
print(np.round(pred))