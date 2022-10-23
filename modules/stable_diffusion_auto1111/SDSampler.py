from abc import ABCMeta, abstractmethod


def setup_img2img_steps(p, steps=None, img2img_fix_steps=False):
    if steps is not None:
        steps = int((steps or p.steps) / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = p.steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc

class SDSampler(metaclass=ABCMeta):
    def __init__(self, plugin):
        self.plugin = plugin
        self.p = None # State parameters, set upon using the sampler
