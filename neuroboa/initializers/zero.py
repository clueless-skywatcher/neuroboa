import numpy as np

from neuroboa.initializers.base_init import BaseInitializer


class ZeroInitializer(BaseInitializer):
    def initialize(self):
        return np.zeros((self.fan_in, self.fan_out))