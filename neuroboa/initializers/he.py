from .base_init import BaseInitializer

import numpy as np

class HeNormalInitializer(BaseInitializer):
    def initialize(self):
        sigma = np.sqrt(2 / self.fan_in)
        return np.random.normal(0, sigma, (self.fan_in, self.fan_out))

class HeUniformInitializer(BaseInitializer):
    def initialize(self):
        factor = np.sqrt(6 / self.fan_in)
        return np.random.uniform(-factor, factor, (self.fan_in, self.fan_out))