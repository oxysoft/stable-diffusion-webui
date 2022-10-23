import os
import signal
import sys
import shlex

from core import cmdargs, paths
from core.paths import repodir, rootdir
from core.installing import is_installed, run, git, git_clone, run_python, run_pip

def print_info():
    try:
        commit = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        commit = "<none>"
    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")


def install_core():
    """ Install all core requirements """
    
    # Creates required download directory
    os.makedirs(repodir, exist_ok=True)
    
    # Setting up installation variables
    requirements_file = os.environ.get('REQS_FILE', "requirements.txt")
    taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
    
    # Updates virtual environment with required modules
    run_pip("install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113", "Pre-Compiled CUDA torch/torchvision")
    run_pip("install --upgrade pip setuptools", "updated Python libraries")
    run_pip(f"install -r {requirements_file}", "requirements for Web UI")

    if not '--skip-torch-cuda-test' in args:
        run_python("import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'")

    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")

    git_clone("https://github.com/CompVis/taming-transformers.git", repodir / "taming-transformers", "Taming Transformers", taming_transformers_commit_hash)
    print("Installations complete")


def start_webui():
    print(f"Launching Web UI with arguments: {' '.join(sys.argv[1:])}")
    from core import webui
    webui.app.run()


def sigint_handler(sig, frame):
    print(f'Interrupted with signal {sig} in {frame}')
    os._exit(0)


def memory_monitor():
    from core import options, devicelib, memmon
    # TODO: remove options
    # mem_mon = memmon.MemUsageMonitor("MemMon", devicelib.device, None)
    mem_mon = memmon.MemUsageMonitor("MemMon", devicelib.device, options.opts)
    mem_mon.start()


def plugin_handler():
    from core.plugins import broadcast, load_all
    
    # Iterate all directories in paths.repodir
    for d in paths.repodir.iterdir():
        sys.path.insert(0, d.as_posix())

    # TODO: git clone modules from a user list
    load_all(paths.plugindir)
    broadcast("launch")  # doubt we need this
    broadcast("install")

    # Dry run, only install and exit.
    if cargs.dry:
        print("Exiting because of --dry argument")
        exit(0)
        

if __name__ == "__main__":
    """ Installs necessary requirements and launches WebUI """

    # Prepare args for use & CTRL-C handler
    signal.signal(signal.SIGINT, sigint_handler)
    args = shlex.split(os.environ.get('COMMANDLINE_ARGS', ""))
    cargs = cmdargs.parser.parse_args(args)
       
    # Initialize
    print_info() 
    install_core()
    memory_monitor()
    plugin_handler()
    
    # Imports modules after prerequisites are met
    from modules.stable_diffusion.SDJob_txt2img import SDJob_txt2img
    from core import plugins

    # Start server
    default_prompt = "Beautiful painting of an ultra contorted landscape by Greg Ruktowsky, airbrushed"
    plugins.job(SDJob_txt2img(prompt=default_prompt, cfg=7.0, steps=20, sampler='euler-a', ))
    start_webui()
