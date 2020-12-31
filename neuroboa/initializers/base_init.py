class BaseInitializer(object):
    def __init__(self, fan_in = None, fan_out = None):
        self.fan_in = fan_in
        self.fan_out = fan_out

    def initialize(self, w):
        pass
    