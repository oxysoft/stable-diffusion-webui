import argparse
import json
import os
import sys

import gradio as gr
import torch

import devices as devices
# from paths import models_path, script_path, sd_path
from core import paths
from modules.stable_diffusion_auto2222 import errors
from modules.stable_diffusion_auto2222.SDAttention import SDAttention

config = paths.repodir / 'stable_diffusion' / 'configs/stable-diffusion/v1-inference.yaml'
ckpt = "models/Stable-diffusion/sd-v1-4.ckpt"
ckpt_dir = paths.modeldir / 'Stable-diffusion'
embeddings_dir = paths.rootdir / 'embeddings'
hypernetwork_dir = ckpt_dir / 'hypernetworks'
vae_path = None

ckpt_dir = ckpt_dir.as_posix()

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

parser = argparse.ArgumentParser()
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)", default=None)
parser.add_argument("--disable-safe-unpickle", action='store_true', help="disable checking pytorch models for malicious code", default=False)

cmd_opts = parser.parse_args()

# Hardware and optimizations
# ----------------------------------------
weight_load_location = None if lowram else "cpu"

batch_cond_uncond = always_batch_cond_uncond or not (lowvram or medvram)
parallel_processing_allowed = not lowvram and not medvram
xformers_available = False
device = devices.get_optimal_device()
errors.run(devices.enable_tf32, "Enabling TF32")

dtype = torch.float16
dtype_vae = torch.float16

devices.device = device
devices.dtype = device
devices.dtype_vae = device

# Hypernetworks
# ----------------------------------------
os.makedirs(hypernetwork_dir, exist_ok=True)

from hypernetworks import hypernetwork

hypernetworks = hypernetwork.list_hypernetworks(hypernetwork_dir)
loaded_hypernetwork = None


def reload_hypernetworks():
    global hypernetworks

    hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)
    hypernetwork.load_hypernetwork(opts.sd_hypernetwork)


# Interrogate
# ----------------------------------------
# import interrogate
# interrogator = interrogate.InterrogateModels("interrogate")


# Options
# ----------------------------------------
class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh


def options_section(section_identifier, options_dict):
    for k, v in options_dict.items():
        v.section = section_identifier

    return options_dict


options_templates = {}

options_templates.update(options_section(('system', "System"), {
    "memmon_poll_rate"  : OptionInfo(8, "VRAM usage polls per second during generation. Set to 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    "multiple_tqdm"     : OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
}))

options_templates.update(options_section(('training', "Training"), {
    "unload_models_when_training"     : OptionInfo(False, "Move VAE and CLIP to RAM when training hypernetwork. Saves VRAM."),
    "dataset_filename_word_regex"     : OptionInfo("", "Filename word regex"),
    "dataset_filename_join_string"    : OptionInfo(" ", "Filename join string"),
    "training_image_repeats_per_epoch": OptionInfo(1, "Number of repeats for a single input image per epoch; used only for displaying epoch number", gr.Number, {"precision": 0}),
    "training_write_csv_every"        : OptionInfo(500, "Save an csv containing the loss to log directory every N steps, 0 to disable"),
}))

options_templates.update(options_section(('sd', "Stable Diffusion"), {
    "sd_checkpoint_cache"            : OptionInfo(0, "Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_hypernetwork"                : OptionInfo("None", "Hypernetwork", gr.Dropdown, lambda: {"choices": ["None"] + [x for x in hypernetworks.keys()]}, refresh=reload_hypernetworks),
    "sd_hypernetwork_strength"       : OptionInfo(1.0, "Hypernetwork strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.001}),
    "img2img_fix_steps"              : OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies (normally you'd do less with less denoising)."),
    "enable_quantization"            : OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply."),
    "enable_emphasis"                : OptionInfo(True, "Emphasis: use (text) to make model pay more attention to text and [text] to make it pay less attention"),
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    "enable_batch_seeds"             : OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
    "comma_padding_backtrack"        : OptionInfo(20, "Increase coherency by padding from the last comma within n tokens when using more than 75 tokens", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1}),
    "filter_nsfw"                    : OptionInfo(False, "Filter NSFW content"),
    'CLIP_stop_at_last_layers'       : OptionInfo(1, "Stop At last layers of CLIP model", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
}))

options_templates.update(options_section(('interrogate', "Interrogate Options"), {
    "interrogate_keep_models_in_memory"    : OptionInfo(False, "Interrogate: keep models in VRAM"),
    "interrogate_use_builtin_artists"      : OptionInfo(True, "Interrogate: use artists from artists.csv"),
    "interrogate_return_ranks"             : OptionInfo(False, "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."),
    "interrogate_clip_num_beams"           : OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length"          : OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length"          : OptionInfo(48, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit"          : OptionInfo(1500, "CLIP: maximum number of lines in text file (0 = No limit)"),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "Interrogate: deepbooru score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha"                 : OptionInfo(True, "Interrogate: deepbooru sort alphabetically"),
    "deepbooru_use_spaces"                 : OptionInfo(False, "use spaces for tags in deepbooru"),
    "deepbooru_escape"                     : OptionInfo(True, "escape (\\) brackets in deepbooru (so they are used as literal brackets and not for emphasis)"),
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters"), {
    "eta_ddim"            : OptionInfo(0.0, "eta (noise multiplier) for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "eta_ancestral"       : OptionInfo(1.0, "eta (noise multiplier) for ancestral samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "ddim_discretize"     : OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn'             : OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_tmin'              : OptionInfo(0.0, "sigma tmin", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise'             : OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}),
}))


class Options:
    data = None
    data_labels = options_templates
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data:
                self.data[key] = value

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def save(self, filename):
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key, func):
        item = self.data_labels.get(key)
        item.onchange = func

        func()

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        return json.dumps(d)

    def add_option(self, key, info):
        self.data_labels[key] = info

    def reorder(self):
        """reorder settings so that all items related to section always go together"""

        section_ids = {}
        settings_items = self.data_labels.items()
        for k, item in settings_items:
            if item.section not in section_ids:
                section_ids[item.section] = len(section_ids)

        self.data_labels = {k: v for k, v in sorted(settings_items, key=lambda x: section_ids[x[1].section])}


opts = Options()

sd_model = None
clip_model = None
progress_print_out = sys.stdout

# class TotalTQDM:
#     def __init__(self):
#         self._tqdm = None
#
#     def reset(self):
#         self._tqdm = tqdm.tqdm(
#             desc="Total progress",
#             total=state.job_count * state.sampling_steps,
#             position=1,
#             file=progress_print_out
#         )
#
#     def update(self):
#         if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
#             return
#         if self._tqdm is None:
#             self.reset()
#         self._tqdm.update()
#
#     def updateTotal(self, new_total):
#         if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
#             return
#         if self._tqdm is None:
#             self.reset()
#         self._tqdm.total=new_total
#
#     def clear(self):
#         if self._tqdm is not None:
#             self._tqdm.close()
#             self._tqdm = None


# total_tqdm = TotalTQDM()
