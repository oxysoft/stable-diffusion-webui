import torch

from core import devicelib


class HypernetworkModule(torch.nn.Module):
    def __init__(self, dim, state_dict=None):
        super().__init__()

        self.linear1 = torch.nn.Linear(dim, dim * 2)
        self.linear2 = torch.nn.Linear(dim * 2, dim)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)
        else:
            self.linear1.weight.data.normal_(mean=0.0, std=0.01)
            self.linear1.bias.data.zero_()
            self.linear2.weight.data.normal_(mean=0.0, std=0.01)
            self.linear2.bias.data.zero_()

        self.to(devicelib.device)

    def forward(self, x):
        return x + (self.linear2(self.linear1(x)))