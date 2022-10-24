import os
import threading

from core.plugins import Plugin
from modules.stable_diffusion_auto2222 import sd_models, sd_samplers
from modules.stable_diffusion_auto2222.processing import process_images, SDJob_txt
from modules.stable_diffusion_auto2222.sd_models import model_path, discover_models

queue_lock = threading.Lock()

def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


class SDPlugin(Plugin):
    def load(self):
        # modelloader.cleanup_models()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        discover_models()
        # codeformer.setup_model(cmd_opts.codeformer_models_path)
        # gfpgan.setup_model(cmd_opts.gfpgan_models_path)
        # shared.face_restorers.append(modules.face_restoration.FaceRestoration())
        # modelloader.load_upscalers()

        # modules.scripts.load_scripts()

        # shared.args

        sd_models.load_model()
        # shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: sd_models.reload_model_weights(shared.sd_model)))
        # shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
        # shared.opts.onchange("sd_hypernetwork_strength", hypernetworks.hypernetwork.apply_strength)

        sd_samplers.set_samplers()

        print('Refreshing Model List')
        sd_models.discover_models()

    def txt2img(self, prompt:str):
        process_images(SDJob_txt(prompt=prompt))
