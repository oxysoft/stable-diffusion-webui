import os
from pathlib import Path

import torch

from core.cmdargs import cargs
from core.printing import printerr
from modules.StableDiffusion.hypernetwork_module import HypernetworkModule
from modules.StableDiffusion.hypernetwork import Hypernetwork


class HypernetworkLoader:
    def __init__(self):
        self.hypernetworks_info = []
        self.hypernetworks_loaded = []

    def reload_hypernetworks(self):
        os.makedirs(cargs.hypernetwork_dir, exist_ok=True)

        self.hypernetworks_info = self.list_paths(cargs.hypernetwork_dir)
        self.hypernetworks_loaded = self.load_hypernetwork(self.hypernetworks_info[0])


    def load_hypernetwork(self, hn):
        if isinstance(hn, str):
            # TODO is this even necessary
            hn = self.find_closest_hypernetwork_name(hn)
            if hn is None:
                return None

            # Iterate through hypernetworks_info and find by name
            for info in self.hypernetworks_info:
                if info[0] == hn:
                    hn = info
                    break

        if hn is None:
            printerr("No hypernetworks found.")
            return None

        if hn.name is None:
            hn.name = hn.path.stem()

        state_dict = torch.load(hn.filepath, map_location='cpu')

        for size, sd in state_dict.items():
            if type(size) == int:
                hn.layers[size] = (HypernetworkModule(size, sd[0]), HypernetworkModule(size, sd[1]))

        hn.name = state_dict.get('name', hn.name)
        hn.step = state_dict.get('step', 0)
        hn.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        hn.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)

        return hn

    def save(self, hn, path):
        state_dict = {}

        for k, v in hn.layers.items():
            state_dict[k] = (v[0].state_dict(), v[1].state_dict())

        state_dict['step'] = hn.step
        state_dict['name'] = hn.name
        state_dict['sd_checkpoint'] = hn.sd_checkpoint
        state_dict['sd_checkpoint_name'] = hn.sd_checkpoint_name

        torch.save(state_dict, path)


    def list_paths(self, dirpath):
        res = []
        for path in Path(dirpath).rglob('*.pt'):
            res.append(path)
        return res

    def instantiate_paths(self, paths):
        return [Hypernetwork(path) for path in paths]


    def apply_hypernetwork(self, hypernetwork, context, layer=None):
        hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)

        if hypernetwork_layers is None:
            return context, context

        if layer is not None:
            layer.hyper_k = hypernetwork_layers[0]
            layer.hyper_v = hypernetwork_layers[1]

        context_k = hypernetwork_layers[0](context)
        context_v = hypernetwork_layers[1](context)
        return context_k, context_v

    def find_closest_hypernetwork_name(self, search: str):
        if not search:
            return None
        search = search.lower()
        applicable = [name for name in self.hypernetworks_info if search in name.lower()]
        if not applicable:
            return None
        applicable = sorted(applicable, key=lambda name: len(name))
        return applicable[0]