import torch

from core import promptlib
from core.cmdargs import cargs


class CFGDenoiser(torch.nn.Module):
    """
    magic code by katherine crowson
    """

    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.step = 0

    def forward(self, x, sigma, uncond, cond, cond_scale):
        batch_cond_uncond = cargs.always_batch_cond_uncond or not (cargs.lowvram or cargs.medvram)

        conds_list, tensor = promptlib.reconstruct_multicond_batch(cond, self.step)
        uncond = promptlib.reconstruct_cond_batch(uncond, self.step)

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
        sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])

        if tensor.shape[1] == uncond.shape[1]:
            cond_in = torch.cat([tensor, uncond])

            if batch_cond_uncond:
                x_out = self.inner_model(x_in, sigma_in, cond=cond_in)
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=cond_in[a:b])
        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size * 2 if batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])
                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=tensor[a:b])

            x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond=uncond)

        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        self.step += 1

        return denoised