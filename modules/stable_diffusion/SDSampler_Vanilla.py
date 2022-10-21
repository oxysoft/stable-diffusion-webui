import torch

from core import promptlib, devicelib

# from StableDiffusionPlugins_samplers import store_latent, setup_img2img_steps
from SDSampler import SDSampler, setup_img2img_steps


class SDSampler_Vanilla(SDSampler):
    def __init__(self, constructor, plugin):
        super().__init__(constructor, plugin)
        self.sampler = constructor(plugin)
        self.orig_p_sample_ddim = self.sampler.p_sample_ddim if hasattr(self.sampler, 'p_sample_ddim') else self.sampler.p_sample_plms
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.sampler_noises = None
        self.step = 0
        self.eta = 0.0 # 0 to 1, 0.01 step
        self.default_eta = 0.0
        self.config = None

    def number_of_needed_noises(self, p):
        return 0

    def p_sample_ddim_hook(self, x_dec, cond, ts, unconditional_conditioning, *args, **kwargs):
        conds_list, tensor = promptlib.reconstruct_multicond_batch(cond, self.step)
        unconditional_conditioning = promptlib.reconstruct_cond_batch(unconditional_conditioning, self.step)

        assert all([len(conds) == 1 for conds in conds_list]), 'composition via AND is not supported for DDIM/PLMS samplers'
        cond = tensor

        # for DDIM, shapes must match, we can't just process cond and uncond independently;
        # filling unconditional_conditioning with repeats of the last vector to match length is
        # not 100% correct but should work well enough
        if unconditional_conditioning.shape[1] < cond.shape[1]:
            last_vector = unconditional_conditioning[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - unconditional_conditioning.shape[1], 1])
            unconditional_conditioning = torch.hstack([unconditional_conditioning, last_vector_repeated])
        elif unconditional_conditioning.shape[1] > cond.shape[1]:
            unconditional_conditioning = unconditional_conditioning[:, :cond.shape[1]]

        if self.mask is not None:
            img_orig = self.sampler.model.q_sample(self.init_latent, ts)
            x_dec = img_orig * self.mask + self.nmask * x_dec

        res = self.orig_p_sample_ddim(x_dec, cond, ts, unconditional_conditioning=unconditional_conditioning, *args, **kwargs)

        # TODO get store_latent from the plugin itself
        if self.mask is not None:
            self.p.store_latent(self.init_latent * self.mask + self.nmask * res[1])
        else:
            self.p.store_latent(res[1])

        self.step += 1
        return res

    def initialize(self, p):
        super(SDSampler_Vanilla, self).initialize(p)

        self.eta = p.eta

        for fieldname in ['p_sample_ddim', 'p_sample_plms']:
            if hasattr(self.sampler, fieldname):
                setattr(self.sampler, fieldname, self.p_sample_ddim_hook)

        self.mask = p.mask if hasattr(p, 'mask') else None
        self.nmask = p.nmask if hasattr(p, 'nmask') else None

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None):
        steps, t_enc = setup_img2img_steps(p, steps)

        self.initialize(p)

        # existing code fails with certain step counts, like 9
        try:
            self.sampler.make_schedule(ddim_num_steps=steps, ddim_eta=self.p.eta, ddim_discretize=p.ddim_discretize, verbose=False)
        except Exception:
            self.sampler.make_schedule(ddim_num_steps=steps + 1, ddim_eta=self.p.eta, ddim_discretize=p.ddim_discretize, verbose=False)

        x1 = self.sampler.stochastic_encode(x, torch.tensor([t_enc] * int(x.shape[0])).to(devicelib.device), noise=noise)

        self.init_latent = x
        self.step = 0

        samples = self.sampler.decode(x1, conditioning, t_enc, unconditional_guidance_scale=p.cfg, unconditional_conditioning=unconditional_conditioning)

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None):
        self.initialize(p)

        self.init_latent = None
        self.step = 0

        steps = steps or p.steps

        # existing code fails with certain step counts, like 9
        try:
            samples_ddim, _ = self.sampler.sample(S=steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg, unconditional_conditioning=unconditional_conditioning, x_T=x, eta=self.eta)
        except Exception:
            samples_ddim, _ = self.sampler.sample(S=steps + 1, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg, unconditional_conditioning=unconditional_conditioning, x_T=x, eta=self.eta)

        return samples_ddim