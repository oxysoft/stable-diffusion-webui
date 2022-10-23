from abc import abstractmethod

import numpy as np
import torch
from PIL import Image

from core import devicelib
from core.jobs import JobParams
from modules.stable_diffusion_auto1111.SDSubseedParams import SDSubseedParams
from modules.stable_diffusion_auto1111.SDUtil import slerp


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
                 model: str = None,
                 n_iter: int = 1,
                 overlay_images=None,  # TODO do we need this??
                 eta=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.seed: int = seed
        self.subseed = subseed
        self.width: int = width
        self.height: int = height
        self.n_iter: int = n_iter
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
        self.enable_batch_seed = False

        if subseed is None:
            self.subseed = SDSubseedParams()
            self.subseed.seed = -1
            self.subseed.strength = 0
            self.subseed.resize_from_h = 0
            self.subseed.resize_from_w = 0

        # self.prompt_for_display: str = None
        # self.n_iter: int = n_iter
        # self.paste_to = None
        # self.color_corrections = None
        # self.denoising_strength: float = 0
        self.sampler_noise_scheduler_override = None

        self.s_churn = 0.0
        self.s_tmin = 0.0
        self.s_tmax = float('inf')  # not representable as a standard ui option
        self.s_noise = 1.0
        self.eta_noise_seed_delta = 0
        self.eta_ancestral = 1.0

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
            x = self.plugin.model.decode_first_stage(x)

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
            job.image = self.sample_to_image(sample)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, sampler, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        raise NotImplementedError()

    def create_random_tensors(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
        xs = []

        # if we have multiple seeds, this means we are working with batch size>1; this then
        # enables the generation of additional tensors with noise that the sampler will use during its processing.
        # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
        # produce the same images as with two batches [100], [101].
        if p is not None \
                and p.sampler is not None \
                and (len(seeds) > 1 and p.enable_batch_seeds or p.eta_noise_seed_delta > 0):
            sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
        else:
            sampler_noises = None

        for i, seed in enumerate(seeds):
            noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h // 8, seed_resize_from_w // 8)

            subnoise = None
            if subseeds is not None:
                subseed = 0 if i >= len(subseeds) else subseeds[i]

                subnoise = devicelib.randn(subseed, noise_shape)

            # randn results depend on device; gpu and cpu get different results for same seed;
            # the way I see it, it's better to do this on CPU, so that everyone gets same result;
            # but the original script had it like this, so I do not dare change it for now because
            # it will break everyone's seeds.
            noise = devicelib.randn(seed, noise_shape)

            if subnoise is not None:
                noise = slerp(subseed_strength, noise, subnoise)

            if noise_shape != shape:
                x = devicelib.randn(seed, shape)
                dx = (shape[2] - noise_shape[2]) // 2
                dy = (shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty:ty + h, tx:tx + w] = noise[:, dy:dy + h, dx:dx + w]
                noise = x

            if sampler_noises is not None:
                cnt = p.sampler.number_of_needed_noises(p)

                if core.options.opts.eta_noise_seed_delta > 0:
                    torch.manual_seed(seed + core.options.opts.eta_noise_seed_delta)

                for j in range(cnt):
                    sampler_noises[j].append(devicelib.randn_without_seed(tuple(noise_shape)))

            xs.append(noise)

        if sampler_noises is not None:
            p.sampler.sampler_noises = [torch.stack(n).to(devicelib.device) for n in sampler_noises]

        x = torch.stack(xs).to(devicelib.device)

        return x
