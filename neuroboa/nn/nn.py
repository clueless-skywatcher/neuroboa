from neuroboa.losses import MSE
from neuroboa.optims import SGD
from neuroboa.constants import *

import numpy as np

from terminaltables import AsciiTable

from tqdm import trange
from tqdm.notebook import trange as trange_notebook

class NN():
    def __init__(self, layers = []):
        self.layers = layers
        
    def _forward(self, x, training = True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def _backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def add(self, layer):
        if not self.layers and layer.input_shape is None:
            raise Exception("Please specify an input shape for the first layer")
        if self.layers:
            layer.input_shape = self.layers[-1].output_shape()
        if hasattr(layer, "_precompute"):
            layer._precompute()
        self.layers.append(layer)

    def fit(self, X, y, 
        epochs = 1000, 
        batch_size = 2, 
        loss = MSE(), 
        optimizer = SGD(),
        show_progress = NO_PBAR,
        record_epochs = True):

        if self.layers[0].input_shape is None:
            raise Exception("Please specify input_shape for the first layer")

        self.loss = loss
        self.optimizer = optimizer
        
        for i in range(len(self.layers)):
            self.layers[i]._set_optimizer(optimizer)
            if i > 0:
                self.layers[i].input_shape = self.layers[i - 1].output_shape()
            if hasattr(self.layers[i], "_precompute"):
                self.layers[i]._precompute()

        rng = range(epochs)

        if show_progress == TQDM_TERMINAL:
            rng = trange(epochs)
        elif show_progress == TQDM_NOTEBOOK:
            rng = trange_notebook(epochs)
        
        self.loss_list = np.array([], dtype='float32')

        epoch_loss = 0.0

        for epoch in rng:
            batch = np.random.choice(range(len(X)), size = batch_size, replace = True)
            pred = self._forward(X[batch])
            epoch_loss = loss.loss(y[batch], pred)
            grad = loss.gradient(y[batch], pred)
            self._backward(grad)
            self.loss_list = np.append(self.loss_list, epoch_loss)
            if show_progress != NO_PBAR:
                rng.set_description(f"Epoch: {epoch + 1}")
                rng.refresh()
            
    def predict(self, inputs):
        return self._forward(inputs, training = False)

    def get_loss_list(self, plot = True):
        if plot:
            print("Plot: True")
        return self.loss_list

    def overview(self):
        print("Layers:")
        overview = []
        for layer in self.layers:
            overview.append(layer._overview())

        table = AsciiTable(overview)
        table.inner_heading_row_border = False

        print(table.table)






        
