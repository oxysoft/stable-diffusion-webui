import torch


class TorchHijack:
    def __init__(self, kdiff_sampler):
        self.kdiff_sampler = kdiff_sampler

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.kdiff_sampler.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))