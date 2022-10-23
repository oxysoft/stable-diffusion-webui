import math

import numpy as np
import torch
from PIL.Image import Image

from core import imagelib, devicelib
from core.options import opts
from modules.stable_diffusion_auto1111.SDConstants import opt_C, opt_f
from modules.stable_diffusion_auto1111.SDJob import SDJob


class SDJob_txt2img(SDJob):
    def get_plugin_impl(self):
        return 'stable_diffusion', 'txt2img'

    def __init__(self, enable_hr=False, scale_latent=True, denoising_strength=0.75, sampler='euler-a', **kwargs):
        # TODO take in input images
        super().__init__(**kwargs)
        self.sampler = sampler
        self.firstphase_width = 0
        self.firstphase_height = 0
        self.firstphase_width_truncated = 0
        self.firstphase_height_truncated = 0
        self.enable_hr = enable_hr  # TODO what is hr
        self.scale_latent = scale_latent
        self.denoising_strength = denoising_strength

    def init(self, all_prompts, all_seeds, all_subseeds):
        # TODO what is enable_hr ...
        if self.enable_hr:
            # if state.job_count == -1:
            #     state.job_count = self.n_iter * 2
            # else:
            #     state.job_count = state.job_count * 2

            desired_pixel_count = 512 * 512
            actual_pixel_count = self.width * self.height
            scale = math.sqrt(desired_pixel_count / actual_pixel_count)

            self.firstphase_width = math.ceil(scale * self.width / 64) * 64
            self.firstphase_height = math.ceil(scale * self.height / 64) * 64
            self.firstphase_width_truncated = int(scale * self.width)
            self.firstphase_height_truncated = int(scale * self.height)

    def sample(self, sampler, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        # TODO what is enable_hr ...
        if not self.enable_hr:
            x = self.create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f],
                                           seeds=seeds,
                                           subseeds=subseeds,
                                           subseed_strength=self.subseed.strength,
                                           seed_resize_from_h=self.subseed.resize_from_h,
                                           seed_resize_from_w=self.subseed.resize_from_w,
                                           p=self)
            samples = sampler.sample(self, x, conditioning, unconditional_conditioning)
            return samples

        x = self.create_random_tensors([opt_C, self.firstphase_height // opt_f, self.firstphase_width // opt_f],
                                       seeds=seeds,
                                       subseeds=subseeds,
                                       subseed_strength=self.subseed.strength,
                                       seed_resize_from_h=self.subseed.resize_from_h,
                                       seed_resize_from_w=self.subseed.resize_from_w,
                                       p=self)
        samples = sampler.sample(self, x, conditioning, unconditional_conditioning)

        truncate_x = (self.firstphase_width - self.firstphase_width_truncated) // opt_f
        truncate_y = (self.firstphase_height - self.firstphase_height_truncated) // opt_f

        samples = samples[:, :, truncate_y // 2:samples.shape[2] - truncate_y // 2, truncate_x // 2:samples.shape[3] - truncate_x // 2]

        if self.scale_latent:
            samples = torch.nn.functional.interpolate(samples, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")
        else:
            decoded_samples = self.plugin().decode_first_stage(self.plugin.model, samples)  # TODO it's in the plugin

            if opts.upscaler_for_img2img is None or opts.upscaler_for_img2img == "None":
                decoded_samples = torch.nn.functional.interpolate(decoded_samples, size=(self.height, self.width), mode="bilinear")
            else:
                lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

                batch_images = []
                for i, x_sample in enumerate(lowres_samples):
                    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(np.uint8)
                    image = Image.fromarray(x_sample)
                    image = imagelib.resize_image(0, image, self.width, self.height)
                    image = np.array(image).astype(np.float32) / 255.0
                    image = np.moveaxis(image, 2, 0)
                    batch_images.append(image)

                decoded_samples = torch.from_numpy(np.array(batch_images))
                decoded_samples = decoded_samples.to(devicelib.device)
                decoded_samples = 2. * decoded_samples - 1.

            samples = self.plugin.model.get_first_stage_encoding(self.plugin.model.encode_first_stage(decoded_samples))

        noise = self.create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, seed_resize_from_h=self.subseed.resize_from_h, seed_resize_from_w=self.subseed.resize_from_w, p=self)

        # GC now before running the next img2img to prevent running out of memory
        x = None
        devicelib.torch_gc()

        samples = sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.steps)

        return samples
