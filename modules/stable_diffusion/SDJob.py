from abc import abstractmethod

import numpy as np
import torch
from PIL.Image import Image

from core import devicelib
from core.jobs import JobParams
from modules.stable_diffusion.SDSubseedParams import SDSubseedParams

class SDJob(JobParams):
    def get_plugin_impl(self):
        raise NotImplementedError()

    def __init__(self,
                 prompt: str = "",
                 promptneg="",
                 seed: int = -1,
                 subseed: SDSubseedParams = None,
                 steps: int = 22,
                 cfg: float = 7,
                 sampler="euler-a",
                 ddim_discretize: bool = 'uniform',  # or quad
                 width=512,
                 height=512,
                 batch_size=1,
                 tiling=False,
                 quantize=False,
                 model:str=None,
                 n_iter: int = 1,
                 overlay_images=None,  # TODO do we need this??
                 eta=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.seed: int = seed
        self.subseed = subseed
        self.width: int = width
        self.height: int = height
        self.n_iter: int = 0
        self.batch_size: int = batch_size
        self.overlay_images = overlay_images
        self.steps: int = steps
        self.prompt: str = prompt
        self.promptneg: str = promptneg or ""
        self.cfg: float = cfg
        self.eta = eta
        self.eta_ancestral = eta
        self.tiling: bool = tiling
        self.sampler: str = sampler
        self.ddim_discretize: bool = ddim_discretize
        self.quantize: bool = quantize
        self.model = model

        if subseed is None:
            subseed = SDSubseedParams()
            subseed.seed = -1
            subseed.strength = 0
            subseed.resize_from_h = 0
            subseed.resize_from_w = 0


        # self.prompt_for_display: str = None
        # self.n_iter: int = n_iter
        # self.paste_to = None
        # self.color_corrections = None
        # self.denoising_strength: float = 0
        # self.sampler_noise_scheduler_override = None

        self.s_churn = 0.0
        self.s_tmin = 0.0
        self.s_tmax = float('inf')  # not representable as a standard ui option
        self.s_noise = 1.0

    def on_start(self, job):
        # TODO unload and load the right model to use if we specified one

        # if isinstance(self.model, str):
        #     name = self.model
        #     self.model = self.plugin.checkpoints.get(name)
        #     if self.model is None:
        #         raise Exception(f"Model {name} not found, using default model instead.")

        self.model = job.plugin.model

    def decode_first_stage(self, x):
        with devicelib.autocast(enable=x.dtype == devicelib.dtype_vae):
            x = self.model.decode_first_stage(x)

        return x

    def sample_to_image(self, samples):
        # TODO check that the right model is loaded and were able to do this (we cant do this during generation when lowvram and medvram are enabled together)
        x_sample = self.decode_first_stage(samples[0:1])[0]
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample = x_sample.astype(np.uint8)

        return Image.fromarray(x_sample)

    def store_latent(self, sample, job):
        job.latent = sample
        if not self.plugin.allow_parallel_processing():
            job.image = self.plugin.sample_to_image(sample)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        raise NotImplementedError()