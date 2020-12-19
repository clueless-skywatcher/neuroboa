from neuroboa.nn import NN
from neuroboa.layers import Dense, Activation
from neuroboa.functions import ReLU, Tanh, Sigmoid
from neuroboa.optims import SGD, Adam, Adagrad
from neuroboa.losses import BinaryCrossEntropy, MSE
from neuroboa.constants import TQDM_TERMINAL

import numpy as np

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

outputs = np.array([
    [1],
    [0],
    [0],
    [1]
])

nn = NN()
nn.add(Dense(50, input_shape = (2,)))
nn.add(Activation(Tanh()))
nn.add(Dense(1))
nn.add(Activation(Sigmoid()))

nn.overview()

nn.fit(inputs, outputs, 
    batch_size = 32, 
    optimizer = Adagrad(), 
    loss = BinaryCrossEntropy(), 
    epochs = 2000,
    show_progress = TQDM_TERMINAL)

pred = nn.predict(np.array([[1, 0], [0, 0], [1, 1], [0, 1]]))
print(np.round(pred))