import platform

from core.installing import run_pip
from core.plugins import Plugin


class XFormersPlugin(Plugin):
    def install(self, args):
        if platform.python_version().startswith("3.10"):
            if platform.system() == "Windows":
                run_pip("install https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/c/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl", "xformers")
            elif platform.system() == "Linux":
                run_pip("install xformers", "xformers")