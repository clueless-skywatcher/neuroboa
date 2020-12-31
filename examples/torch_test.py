import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from neuroboa.nn import NN
from neuroboa.layers import Dense, Activation
from neuroboa.functions import ReLU, Softmax, Tanh
from neuroboa.losses import BinaryCrossEntropy, SoftmaxCrossEntropy
from neuroboa.optims import Adam
from neuroboa.constants import TQDM_TERMINAL

from sklearn.preprocessing import OneHotEncoder

import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = False)

test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = False)

nn = NN([
    Dense(170, input_shape = (784,)),
    Activation(Tanh()),
    Dense(10),
    Activation(Softmax())
])

X_tr = train_dataset.data.numpy()
X_t = test_dataset.data.numpy()

X_tr = X_tr.reshape(X_tr.shape[0], -1)
X_t = X_t.reshape(X_t.shape[0], -1)

one_hot = OneHotEncoder()

y_tr = train_dataset.targets.numpy().reshape(-1, 1)
y_t = test_dataset.targets.numpy().reshape(-1, 1)

y_tr1 = one_hot.fit_transform(y_tr).toarray()
y_t1 = one_hot.fit_transform(y_t).toarray()

nn.fit(X_tr, y_tr1, epochs = 500, batch_size = 64, loss = BinaryCrossEntropy(), optimizer = Adam(lr = 0.001), show_progress = TQDM_TERMINAL)
preds = np.round(nn.predict(X_t))

total = len(preds)

correct = 0

for pred, y in zip(preds, y_t1):
    if pred.argmax() == y.argmax():
        correct += 1

print(f"Accuracy: {float(correct) * 100 / total}%")