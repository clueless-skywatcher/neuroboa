from neuroboa.functions import Softmax

import numpy as np

softmax = Softmax()
print(softmax(np.array([5, 3, 2])))