from core.jobs import JobParams
from modules.StableDiffusion.jobs import SubseedParams


class SDJob(JobParams):

    def __init__(self,
                 plugin=None,
                 prompt: str = "",
                 promptneg="",
                 seed: int = -1,
                 subseed: SubseedParams = None,
                 steps: int = 22,
                 cfg: float = 7,
                 sampler="euler-a",
                 ddim_discretize: bool = 'uniform', # or quad
                 width=512,
                 height=512,
                 batch_size=1,
                 tiling=False,
                 quantize=False,
                 n_iter: int = 1,
                 overlay_images=None, # TODO do we need this??
                 eta=None,
                 **kwargs):
        super().__init__(**kwargs)
        if subseed is None:
            subseed = SubseedParams()
            subseed.seed = -1
            subseed.strength = 0
            subseed.resize_from_h = 0
            subseed.resize_from_w = 0

        self.plugin = plugin
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
        self.tiling: bool = tiling
        self.sampler: str = sampler
        self.ddim_discretize: bool = ddim_discretize
        self.quantize: bool = quantize

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

    def model(self):
        return self.plugin.model

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        raise NotImplementedError()