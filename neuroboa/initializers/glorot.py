from .base_init import BaseInitializer

import numpy as np

class GlorotNormalInitializer(BaseInitializer):
    def initialize(self):
        sigma = np.sqrt(2 / (self.fan_in + self.fan_out))
        return np.random.normal(0, sigma, (self.fan_in, self.fan_out))

class GlorotUniformInitializer(BaseInitializer):
    def initialize(self):
        factor = np.sqrt(6 / (self.fan_in + self.fan_out))
        return np.random.uniform(-factor, factor, (self.fan_in, self.fan_out))