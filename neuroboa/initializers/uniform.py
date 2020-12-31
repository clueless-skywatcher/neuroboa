from .base_init import BaseInitializer

import numpy as np

class UniformInitializer(BaseInitializer):
    def initialize(self):
        factor = 1 / np.sqrt(self.fan_in)

        return np.random.uniform(-factor, factor, (self.fan_in, self.fan_out))