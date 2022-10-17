# this scripts installs necessary requirements and launches main program in webui.py
import os
import sys
import shlex

import plugins
from paths import dir_repos
from api import is_installed, run, git, git_clone, run_python, run_pip, repo_dir, python

from paths import script_path
import webui

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

    os.makedirs(dir_repos, exist_ok=True)

    if not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")

    if not '--skip-torch-cuda-test' in args:
        run_python("import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'")

    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")

    git_clone("https://github.com/CompVis/taming-transformers.git", repo_dir('taming-transformers'), "Taming Transformers", taming_transformers_commit_hash)

def install_webui():
    run_pip(f"install -r {requirements_file}", "requirements for Web UI")

def start_webui():
    print(f"Launching Web UI with arguments: {' '.join(sys.argv[1:])}")
    webui.webui()


if __name__ == "__main__":
    args = shlex.split(commandline_args)
    sys.argv += args

    print_info()
    install_core()

    if "--exit" in args:
        print("Exiting because of --exit argument")
        exit(0)

    # Load and install plugins
    # TODO git clone plugins from a user list
    plugins.reload(os.path.join(script_path, "scripts"))
    plugins.dispatch.install(args)

    # Web UI
    # TODO add a switch to disable webui
    install_webui()
    start_webui()
