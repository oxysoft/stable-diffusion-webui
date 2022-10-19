import json

from core.options import opts
from modules.StableDiffusion.jobs import create_infotext
from modules.StableDiffusion.SDJob import SDJob


class JobResult:
    def __init__(self, p: SDJob, outputs: list, info="", all_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None):
        self.images = outputs
        self.info = info
        # self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.clip_skip = opts.CLIP_stop_at_last_layers  # TODO wat no need to cache

        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0])
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1

        self.all_prompts = all_prompts or [self.prompt]
        self.all_seeds = all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]

    def js(self):
        obj = {
            "prompt"                 : self.prompt,
            "all_prompts"            : self.all_prompts,
            "negative_prompt"        : self.negative_prompt,
            "seed"                   : self.seed,
            "all_seeds"              : self.all_seeds,
            "subseed"                : self.subseed,
            "all_subseeds"           : self.all_subseeds,
            "subseed_strength"       : self.subseed_strength,
            "width"                  : self.width,
            "height"                 : self.height,
            "sampler_index"          : self.sampler_index,
            "sampler"                : self.sampler,
            "cfg_scale"              : self.cfg_scale,
            "steps"                  : self.steps,
            "batch_size"             : self.batch_size,
            "restore_faces"          : self.restore_faces,
            "face_restoration_model" : self.face_restoration_model,
            "sd_model_hash"          : self.sd_model_hash,
            "seed_resize_from_w"     : self.seed_resize_from_w,
            "seed_resize_from_h"     : self.seed_resize_from_h,
            "denoising_strength"     : self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image"   : self.index_of_first_image,
            "infotexts"              : self.infotexts,
            "styles"                 : self.styles,
            "clip_skip"              : self.clip_skip,
        }

        return json.dumps(obj)

    def infotext(self, p: SDJob, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)