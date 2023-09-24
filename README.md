# neuroboa
A Deep Learning library from scratch in Python and NumPy

Inspired by (Please go check out these repos as well):
- Keras
- Joel Grus' Neural Network Library: https://github.com/joelgrus/joelnet/
- ML from Scratch: https://github.com/eriklindernoren/ML-From-Scratch/

Requires:
- NumPy - For matrix calculations
- Tqdm - For showing progress in training the network

Some results that have been found:
- Trained on MNIST handwritten digits
  - Optimizer: Adam with learning rate = 0.001
  - Loss: Binary Cross Entropy
  - Layers: Dense with 50 neurons --> ReLU --> Dense with 10 neurons --> Softmax
  - Batch size = 64
  - Epochs = 1000
  - Accuracy in the 75-80% range (sometimes > 80%)
- Trained on Pima Indians dataset
  - Optimizer: Adam with learning rate = 0.01
  - Loss: Binary Cross Entropy
  - Layers: Dense with 1000 neurons --> ReLU --> Dense with 100 neurons --> Tanh --> Dense with 500 neurons --> ReLU --> Dense with 2 neurons --> Sigmoid
  - Batch size = 10
  - Epochs = 150
  - Accuracy in the 66-76% range
