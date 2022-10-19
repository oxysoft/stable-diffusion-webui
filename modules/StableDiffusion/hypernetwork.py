from modules.StableDiffusion.hypernetwork_module import HypernetworkModule


class Hypernetwork:
    def __init__(self, path=None, enable_sizes=None):
        self.path = path
        self.id = path.stem
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

        for size in enable_sizes or []:
            self.layers[size] = (HypernetworkModule(size), HypernetworkModule(size))

    def weights(self):
        res = []

        for k, layers in self.layers.items():
            for layer in layers:
                layer.train()
                res += [layer.linear1.weight, layer.linear1.bias, layer.linear2.weight, layer.linear2.bias]

        return res