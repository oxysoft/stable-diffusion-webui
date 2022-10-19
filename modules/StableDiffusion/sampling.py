def setup_img2img_steps(p, steps=None, img2img_fix_steps=False):
    if steps is not None:
        steps = int((steps or p.steps) / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = p.steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc

class StableDiffusionSampler:
    def __init__(self, p, plugin):
        self.p = p
        self.plugin = plugin