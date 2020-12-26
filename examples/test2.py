from keras.metrics import binary_crossentropy
from neuroboa.losses import BinaryCrossEntropy

import numpy as np

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

print(binary_crossentropy(y_true, y_pred).numpy())
print(BinaryCrossEntropy().loss(np.array(y_true), np.array(y_pred)))