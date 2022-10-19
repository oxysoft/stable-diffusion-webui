import inspect

import torch
import tqdm
from tqdm import tqdm

# from StableDiffusionPlugins_samplers import sampler_extra_params, setup_img2img_steps
from sampling import StableDiffusionSampler, setup_img2img_steps
from sampling_cfgdenoiser import CFGDenoiser
from sampling_torchhijack import TorchHijack

from core import devicelib
from core.options import opts
from core.cmdargs import cargs
from core.printing import progress_print_out
import k_diffusion.sampling

sampler_extra_params = {
    'sample_euler': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_heun' : ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_2': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
}


def extended_trange(job, sampler, count, *args, **kwargs):
    seq = range(count) if cargs.disable_console_progressbars else tqdm.trange(count, *args, desc=job, file=progress_print_out, **kwargs)

    for x in seq:
        if job.aborted: break
        if sampler.stop_at is not None and x > sampler.stop_at:
            break

        yield x

        job.update_step()


class KDiffusionSampler(StableDiffusionSampler):
    def __init__(self, funcname, p, plugin):
        super().__init__(p, plugin)
        self.model_wrap = k_diffusion.external.CompVisDenoiser(p.model(), quantize=p.quantize)
        self.p = p
        self.funcname = funcname
        self.kfunc = getattr(k_diffusion.sampling, self.funcname) # The k-diffusion function name to call
        self.extra_params = sampler_extra_params.get(funcname, []) # Extra params to pass to the k-diffusion function
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.sampler_noise_index = 0
        self.stop_at = None
        self.eta = None
        self.default_eta = 1.0
        self.config = None

    def callback_state(self, d):
        self.p.model().store_latent(d["denoised"])

    def number_of_needed_noises(self, p):
        return p.steps

    def randn_like(self, x):
        noise = self.sampler_noises[self.sampler_noise_index] if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises) else None

        if noise is not None and x.shape == noise.shape:
            res = noise
        else:
            res = torch.randn_like(x)

        self.sampler_noise_index += 1
        return res

    def initialize(self, p):
        self.model_wrap_cfg.mask = p.mask if hasattr(p, 'mask') else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, 'nmask') else None
        self.model_wrap.step = 0
        self.sampler_noise_index = 0
        self.eta = p.eta or opts.eta_ancestral

        if hasattr(k_diffusion.sampling, 'trange'):
            k_diffusion.sampling.trange = lambda *args, **kwargs: extended_trange(self, *args, **kwargs)

        if self.sampler_noises is not None:
            k_diffusion.sampling.torch = TorchHijack(self)

        # Pick extra params to send to the sampler and what the fuck is going on here
        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if hasattr(p, param_name) and param_name in inspect.signature(self.kfunc).parameters:
                extra_params_kwargs[param_name] = getattr(p, param_name)

        if 'eta' in inspect.signature(self.kfunc).parameters:
            extra_params_kwargs['eta'] = self.eta

        return extra_params_kwargs

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None):
        steps = steps or p.steps

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif self.config is not None and self.config.options.get('scheduler', None) == 'karras':
            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=0.1, sigma_max=10, device=devicelib.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        x = x * sigmas[0]

        extra_params_kwargs = self.initialize(p)
        if 'sigma_min' in inspect.signature(self.kfunc).parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()
            if 'n' in inspect.signature(self.kfunc).parameters:
                extra_params_kwargs['n'] = steps
        else:
            extra_params_kwargs['sigmas'] = sigmas
        samples = self.kfunc(self.model_wrap_cfg, x, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg}, disable=False, callback=self.callback_state, **extra_params_kwargs)
        return samples

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None):
        steps, t_enc = setup_img2img_steps(p, steps)

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif self.config is not None and self.config.options.get('scheduler', None) == 'karras':
            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=0.1, sigma_max=10, device=devicelib.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        sigma_sched = sigmas[steps - t_enc - 1:]
        xi = x + noise * sigma_sched[0]

        extra_params_kwargs = self.initialize(p)
        if 'sigma_min' in inspect.signature(self.kfunc).parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs['sigma_min'] = sigma_sched[-2]
        if 'sigma_max' in inspect.signature(self.kfunc).parameters:
            extra_params_kwargs['sigma_max'] = sigma_sched[0]
        if 'n' in inspect.signature(self.kfunc).parameters:
            extra_params_kwargs['n'] = len(sigma_sched) - 1
        if 'sigma_sched' in inspect.signature(self.kfunc).parameters:
            extra_params_kwargs['sigma_sched'] = sigma_sched
        if 'sigmas' in inspect.signature(self.kfunc).parameters:
            extra_params_kwargs['sigmas'] = sigma_sched

        self.model_wrap_cfg.init_latent = x

        return self.kfunc(self.model_wrap_cfg, xi, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg}, disable=False, callback=self.callback_state, **extra_params_kwargs)
