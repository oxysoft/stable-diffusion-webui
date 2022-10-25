import argparse
import os
import sys

import devices as devices
# from paths import models_path, script_path, sd_path
from core.jobs import JobData, JobParams
from modules.stable_diffusion_auto1111 import sd_hypernetwork, safe
from modules.stable_diffusion_auto1111.SDAttention import SDAttention
from modules.stable_diffusion_auto1111.options import opts

# Constants
from modules.stable_diffusion_auto1111.sd_paths import ckpt_dir, hypernetwork_dir

# Options
attention = SDAttention.SPLIT_DOGGETT
lowvram = False
medvram = True
lowram = False
precision = "full"
no_half = True
opt_channelslast = False
always_batch_cond_uncond = False  # disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram
xformers = False
force_enable_xformers = False
use_cpu = False
batch_cond_uncond = always_batch_cond_uncond or not (lowvram or medvram)

# Arguments
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)", default=None)
cmd_opts = parser.parse_args()

# Hardware and optimizations
# ----------------------------------------
weight_load_location = None if lowram else "cpu"

parallel_processing_allowed = not lowvram and not medvram
safe.run(devices.enable_tf32, "Enabling TF32")
devices.set(devices.get_optimal_device(), 'half')

# Hypernetworks
# ----------------------------------------
os.makedirs(hypernetwork_dir, exist_ok=True)

def reload_hypernetworks():
    global hypernetworks

    hypernetworks = hypernetwork.discover_hypernetworks(cmd_opts.hypernetwork_dir)
    hypernetwork.load_hypernetwork(opts.sd_hypernetwork)


# Interrogate
# ----------------------------------------
# import interrogate
# interrogator = interrogate.InterrogateModels("interrogate")

sdmodel = None
clip_model = None
progress_print_out = sys.stdout
import os
import threading

from core.plugins import Plugin
from modules.stable_diffusion_auto1111 import sd_models, sd_samplers
from modules.stable_diffusion_auto1111.SDJob import process_images, SDJob_txt
from modules.stable_diffusion_auto1111.sd_models import model_path, discover_models

queue_lock = threading.Lock()


class SDPlugin(Plugin):
    def title(self):
        return "Stable Diffusion AUTO1111"

    def load(self):
        # modelloader.cleanup_models()
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        discover_models()

        # codeformer.setup_model(cmd_opts.codeformer_models_path)
        # gfpgan.setup_model(cmd_opts.gfpgan_models_path)
        # SDPlugin.face_restorers.append(modules.face_restoration.FaceRestoration())
        # modelloader.load_upscalers()

        # modules.scripts.load_scripts()

        sd_models.load_model()
        sd_models.discover_models()

        # SDPlugin.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: sd_models.reload_model_weights(SDPlugin.sd_model)))
        # SDPlugin.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: hypernetworks.hypernetwork.load_hypernetwork(SDPlugin.opts.sd_hypernetwork)))
        # SDPlugin.opts.onchange("sd_hypernetwork_strength", hypernetworks.hypernetwork.apply_strength)

        print('Refreshing Model List')

    def txt2img(self, prompt: str):
        process_images(SDJob_txt(prompt=prompt))