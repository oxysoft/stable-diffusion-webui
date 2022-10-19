# from modules.StableDiffusion.util import *
import core.plugins

sd = core.plugins.get("stablediffusion")


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments, iteration=0, position_in_batch=0):
    # index = position_in_batch + iteration * p.batch_size
    #
    # clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)
    #
    # generation_params = {
    #     "Steps"                  : p.steps,
    #     "Sampler"                : get_correct_sampler(p)[p.sampler_index].name,
    #     "CFG scale"              : p.cfg,
    #     "Seed"                   : all_seeds[index],
    #     "Face restoration"       : (opts.face_restoration_model if p.restore_faces else None),
    #     "Size"                   : f"{p.width}x{p.height}",
    #     "Model hash"             : getattr(p, 'sd_model_hash', None if not opts.add_model_hash_to_info or not sd.model.sd_model_hash else sd.model.sd_model_hash),
    #     "Model"                  : (None if not opts.add_model_name_to_info or not sd.model.sd_checkpoint_info.model_name else sd.model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')),
    #     "Hypernet"               : (None if hypernetworks_loaded is None else hypernetworks_loaded.name.replace(',', '').replace(':', '')),
    #     "Batch size"             : (None if p.batch_size < 2 else p.batch_size),
    #     "Batch pos"              : (None if p.batch_size < 2 else position_in_batch),
    #     "Variation seed"         : (None if p.subseed_pow == 0 else all_subseeds[index]),
    #     "Variation seed strength": (None if p.subseed_pow == 0 else p.subseed_pow),
    #     "Seed resize from"       : (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
    #     "Denoising strength"     : getattr(p, 'denoising_strength', None),
    #     "Eta"                    : (None if p.sampler is None or p.sampler.eta == p.sampler.default_eta else p.sampler.eta),
    #     "Clip skip"              : None if clip_skip <= 1 else clip_skip,
    #     "ENSD"                   : None if opts.eta_noise_seed_delta == 0 else opts.eta_noise_seed_delta,
    # }
    #
    # generation_params.update(p.extra_generation_params)
    #
    # generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
    #
    # negative_prompt_text = "\nNegative prompt: " + p.promptneg if p.promptneg else ""
    #
    # return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip()
    return "no info!"


class SubseedParams:
    def __init__(self,
                 subseed=-1,
                 subseed_strength=0,
                 seed_resize_from_h=-1,
                 seed_resize_from_w=-1):
        self.seed: int = subseed
        self.strength: float = subseed_strength
        self.resize_from_h: int = seed_resize_from_h
        self.resize_from_w: int = seed_resize_from_w


