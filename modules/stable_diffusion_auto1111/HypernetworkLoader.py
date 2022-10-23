import os
from pathlib import Path

import torch

from core.cmdargs import cargs
from core.printing import printerr
from CheckpointLoader import CheckpointLoader
from HypernetworkModule import HypernetworkModule
from Hypernetwork import Hypernetwork


class HypernetworkLoader(CheckpointLoader):
    def load_hypernetwork(self, info):
        if isinstance(info, str):
            info = self.get(info)
            if info is None:
                raise ValueError(f'Hypernetwork {info} not found')

        state_dict = torch.load(info.filepath, map_location='cpu')

        for size, sd in state_dict.items():
            if type(size) == int:
                info.layers[size] = (HypernetworkModule(size, sd[0]), HypernetworkModule(size, sd[1]))

        info.name = state_dict.get('name', info.name)
        info.step = state_dict.get('step', 0)
        info.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        info.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)

        return info

    def attention_CrossAttention_forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = np.default(context, x)

        context_k, context_v = self.apply_hypernetwork(hypernetworks_loaded, context, self)
        k = self.to_k(context_k)
        v = self.to_v(context_v)

        q, k, v = map(lambda t: np.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = np.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = np.rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = np.repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = np.einsum('b i j, b j d -> b i d', attn, v)
        out = np.rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

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