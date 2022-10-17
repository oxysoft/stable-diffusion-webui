import os
import time
import importlib
import signal
import threading

from fastapi.middleware.gzip import GZipMiddleware

import modules.extras
import modules.face_restoration
import modules.img2img

import modules.lowvram
import plugins.StableDiffusionPlugin_hijack
import shared as shared
import modules.txt2img

import ui.ui
from modules import devices
import plugins.StableDiffusionPlugin_hypernetworks
from shared import cmd_opts
from plugins import plugins

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    def f(*args, **kwargs):
        devices.torch_gc()

        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.job_timestamp = shared.state.get_job_timestamp()
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0
        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.textinfo = None

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        devices.torch_gc()

        return res

    return ui.ui.wrap_gradio_call(f, extra_outputs=extra_outputs)


def initialize():

def webui():
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: plugins.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: plugins.StableDiffusionPlugin_hypernetworks.load_hypernetwork(shared.opts.sd_hypernetwork)))

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    while 1:
        demo = ui.ui.create_ui(wrap_gradio_gpu_call=wrap_gradio_gpu_call)

        app, local_url, share_url = demo.launch(
                share=cmd_opts.share,
                server_name="0.0.0.0" if cmd_opts.listen else None,
                server_port=cmd_opts.port,
                debug=cmd_opts.gradio_debug,
                auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
                inbrowser=cmd_opts.autolaunch,
                prevent_thread_lock=True)

        app.add_middleware(GZipMiddleware, minimum_size=1000)

        while 1:
            time.sleep(0.5)
            if getattr(demo, 'do_restart', False):
                time.sleep(0.5)
                demo.close()
                time.sleep(0.5)
                break

        print('Reloading modules: modules.ui')
        importlib.reload(ui.ui)

        print('Restarting Gradio')