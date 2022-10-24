import collections
import os.path
import sys
from collections import namedtuple
from urllib.parse import urlparse

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import shared, modelloader
from core import paths, modellib
from sd_hijack_inpainting import do_inpainting_hijack, should_hijack_inpainting

model_dir = "Stable-diffusion"
model_path = paths.modeldir / model_dir

all_infos = {}
all_loaded = collections.OrderedDict()

vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}

CheckpointInfo = namedtuple("CheckpointInfo", ['filename', 'title', 'hash', 'model_name', 'config'])


def checkpoint_titles():
    return sorted([x.title for x in all_infos.values()])


def discover_models():
    all_infos.clear()
    all_paths = modellib.discover_models(model_dir=model_path, command_path=shared.ckpt_dir, ext_filter=[".ckpt"])

    def modeltitle(path, shorthash):
        abspath = os.path.abspath(path)

        if shared.ckpt_dir is not None and abspath.startswith(shared.ckpt_dir):
            name = abspath.replace(shared.ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(path)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

        return f'{name} [{shorthash}]', shortname

    cmd_ckpt = shared.ckpt
    if os.path.exists(cmd_ckpt):
        h = get_model_hash(cmd_ckpt)
        title, short_model_name = modeltitle(cmd_ckpt, h)
        all_infos[title] = CheckpointInfo(cmd_ckpt, title, h, short_model_name, shared.config)
        shared.opts.data['sd_model_checkpoint'] = title

    for filename in all_paths:
        h = get_model_hash(filename)
        title, short_model_name = modeltitle(filename, h)

        basename, _ = os.path.splitext(filename)
        config = basename + ".yaml"
        if not os.path.exists(config):
            config = shared.config

        all_infos[title] = CheckpointInfo(filename, title, h, short_model_name, config)


def get_closest_by_name(search_name):
    applicable = sorted([info for info in all_infos.values() if search_name in info.title], key=lambda x: len(x.title))
    if len(applicable) > 0:
        return applicable[0]
    return None


def get_model_hash(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def select_checkpoint(path=None):
    info = all_infos.get(path, None)
    if info is not None:
        return info

    if len(all_infos) == 0:
        print(f"No checkpoints found. When searching for checkpoints, looked at:", file=sys.stderr)
        if shared.ckpt is not None:
            print(f" - file {os.path.abspath(shared.ckpt)}", file=sys.stderr)
        print(f" - directory {model_path}", file=sys.stderr)
        if shared.ckpt_dir is not None:
            print(f" - directory {os.path.abspath(shared.ckpt_dir)}", file=sys.stderr)
        print(f"Can't run without a checkpoint. Find and place a .ckpt file into any of those locations. The program will exit.", file=sys.stderr)
        exit(1)

    info = next(iter(all_infos.values()))
    if path is not None:
        print(f"Checkpoint {path} not found; loading fallback {info.title}", file=sys.stderr)

    return info


chckpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.'      : 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.'         : 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    if "state_dict" in pl_sd:
        pl_sd = pl_sd["state_dict"]

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def load_model_weights(model, info):
    path = info.filename
    hash = info.hash

    if info not in all_loaded:
        print(f"Loading weights [{hash}] from {path}")

        pl_sd = torch.load(path, map_location=shared.weight_load_location)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")

        sd = get_state_dict_from_checkpoint(pl_sd)
        missing, extra = model.load_state_dict(sd, strict=False)

        if shared.opt_channelslast:
            model.to(memory_format=torch.channels_last)

        if not shared.no_half:
            model.half()

        shared.dtype = torch.float32 if shared.no_half else torch.float16
        shared.dtype_vae = torch.float32 if shared.no_half or shared.no_half_vae else torch.float16

        vae_file = os.path.splitext(path)[0] + ".vae.pt"

        if not os.path.exists(vae_file) and shared.vae_path is not None:
            vae_file = shared.vae_path

        if os.path.exists(vae_file):
            print(f"Loading VAE weights from: {vae_file}")
            vae_ckpt = torch.load(vae_file, map_location=shared.weight_load_location)
            vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss" and k not in vae_ignore_keys}
            model.first_stage_model.load_state_dict(vae_dict)

        model.first_stage_model.to(shared.dtype_vae)

        all_loaded[info] = model.state_dict().copy()
        while len(all_loaded) > shared.opts.sd_checkpoint_cache:
            all_loaded.popitem(last=False)  # LRU
    else:
        print(f"Loading weights [{hash}] from cache")
        all_loaded.move_to_end(info)
        model.load_state_dict(all_loaded[info])

    model.hash = hash
    model.ckptpath = path
    model.info = info


def load_model(info=None):
    from modules.stable_diffusion_auto2222 import lowvram, sd_hijack
    info = info or select_checkpoint()

    if info.config != shared.config:
        print(f"Loading config from: {info.config}")

    config = OmegaConf.load(info.config)

    if should_hijack_inpainting(info):
        # Hardcoded config for now...
        config.model.target = "ldm.models.diffusion.ddpm.LatentInpaintDiffusion"
        config.model.params.use_ema = False
        config.model.params.conditioning_key = "hybrid"
        config.model.params.unet_config.params.in_channels = 9

        # Create a "fake" config with a different name so that we know to unload it when switching models.
        info = info._replace(config=info.config.replace(".yaml", "-inpainting.yaml"))

    do_inpainting_hijack()
    sdmodel = instantiate_from_config(config.model)
    load_model_weights(sdmodel, info)

    if shared.lowvram or shared.medvram:
        lowvram.setup_for_low_vram(sdmodel, shared.medvram)
    else:
        sdmodel.to(shared.device)

    sd_hijack.model_hijack.hijack(sdmodel)

    sdmodel.eval()
    shared.sd_model = sdmodel

    # script_callbacks.model_loaded_callback(sdmodel)

    print(f"Model loaded.")
    return sdmodel


def reload_model_weights(sdmodel, info=None):
    import lowvram, devices, sd_hijack
    info = info or select_checkpoint()

    if sdmodel.ckptpath == info.filename:
        return

    if sdmodel.info.config != info.config or should_hijack_inpainting(info) != should_hijack_inpainting(sdmodel.info):
        all_loaded.clear()
        load_model(info)
        return shared.sd_model

    if shared.lowvram or shared.medvram:
        lowvram.send_everything_to_cpu()
    else:
        sdmodel.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sdmodel)

    load_model_weights(sdmodel, info)

    sd_hijack.model_hijack.hijack(sdmodel)
    # script_callbacks.model_loaded_callback(sd_model)

    if not shared.lowvram and not shared.medvram:
        sdmodel.to(shared.device)

    print(f"Weights loaded.")
    return sdmodel
