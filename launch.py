# this scripts installs necessary requirements and launches main program in webui.py
import importlib
import os
import signal
import sys
import shlex

from core.paths import repodir
from core.installing import is_installed, run, git, git_clone, run_python, run_pip, repo_dir, python

from core.paths import rootdir
from modules.StableDiffusion.SDJob_txt2img import SDJob_txt2img

taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")


def print_info():
    try:
        commit = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        commit = "<none>"
    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")


def install_core():
    """
    Install all core requirements
    """

    os.makedirs(repodir, exist_ok=True)

    if not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")

    if not '--skip-torch-cuda-test' in args:
        run_python("import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'")

    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")

    git_clone("https://github.com/CompVis/taming-transformers.git", repodir / "taming-transformers", "Taming Transformers", taming_transformers_commit_hash)


def install_webui():
    run_pip(f"install -r {requirements_file}", "requirements for Web UI")


def start_webui():
    print(f"Launching Web UI with arguments: {' '.join(sys.argv[1:])}")
    from core import webui
    webui.app.run()


def sigint_handler(sig, frame):
    print(f'Interrupted with signal {sig} in {frame}')
    os._exit(0)


xformers_available = False

if __name__ == "__main__":
    print_info()

    # Prepare CTRL-C handler
    signal.signal(signal.SIGINT, sigint_handler)
    # Prepare args for use
    args = shlex.split(commandline_args)
    sys.argv += args

    # Prepare plugin system
    from core import plugins, cmdargs, options

    # Load modules
    sys.path.insert(0, (rootdir / "repositories" / "stable-diffusion").as_posix())
    sys.path.insert(0, (rootdir / "repositories" / "k-diffusion").as_posix())
    sys.path.insert(0, (rootdir / "repositories" / "stable-diffusion" / "ldm").as_posix())
    sys.path.insert(0, (rootdir / "modules" / "StableDiffusion").as_posix())

    from StableDiffusionPlugin import StableDiffusionPlugin

    sdplug = StableDiffusionPlugin()

    sdplug.args(cmdargs.parser)
    plugins.dispatch.args(cmdargs.parser)
    cmdargs.cargs = cmdargs.parser.parse_args()

    sdplug.options(options.options_templates)
    options.load()


    # Memory monitor
    from core import options, devicelib, memmon

    mem_mon = memmon.MemUsageMonitor("MemMon", devicelib.device, options.opts)
    mem_mon.start()

    # plugins.load_py(root_path / "modules" / "StableDiffusion" / "StableDiffusionPlugin.py")

    # Installations
    install_core()
    plugins.dispatch.install(args)
    sdplug.install()

    # Dry run, only install stuff and exit.
    if "--dry" in args:
        print("Exiting because of --dry argument")
        exit(0)

    # TODO git clone modules from a user list
    plugins.dispatch.launch(args)

    # sdplug.init()
    sdplug.load()
    sdplug.txt2img(
            SDJob_txt2img(prompt="",
                          cfg=7.75,
                          steps=22,
                          sampler='euler-a',
                          ))

    install_webui()
    start_webui()
