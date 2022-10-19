import contextlib

import torch

from core import printing
from core.cmdargs import cargs


def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if has_mps:
        return torch.device("mps")

    return cpu


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    return torch.randn(shape, device=device)


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or cargs.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


# has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
has_mps = getattr(torch, 'has_mps', False)
cpu = torch.device("cpu")
printing.run(enable_tf32, "Enabling TF32")

dtype = torch.float16
dtype_vae = torch.float16

# TODO figure out a device assignment token or some crap that modules can use

# device = device_gfpgan = device_bsrgan = device_esrgan = device_scunet = device_codeformer = get_optimal_device()
# device, \
# device_gfpgan, \
# device_bsrgan, \
# device_esrgan, \
# device_scunet, \
# device_codeformer = \
#     (cpu if x in cargs.use_cpu else get_optimal_device()
#      for x
#      in ['StableDiffusion', 'GFPGAN', 'BSRGAN', 'ESRGAN', 'SCUNet', 'CodeFormer'])

# device = device

device = get_optimal_device()
