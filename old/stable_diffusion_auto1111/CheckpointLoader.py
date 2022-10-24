from typing import Any

from core import paths
from core.printing import printerr
from CheckpointInfo import CheckpointInfo
from modules.stable_diffusion_auto1111.CheckpointInfo import CheckpointInfo

class CheckpointLoader:
    """
    A repository of models in installation_dir/models/<subdirname>
    Automatically detects all models in the directory and provides a list of CheckpointInfo
    A default model ID can be given to use as a default when none is specified by the user configuration.
    """

    def __init__(self, subdirname, config, defaults=None, extensions=None):
        if extensions is None:
            extensions = ['ckpt', 'pt']
        if defaults is None:
            defaults = ["model"]

        self.all = []
        self.config = config
        self.dirpath = paths.modeldir / subdirname
        self.default_ids = defaults
        self.extensions = extensions

        self.reload()

    def get(self, id) -> CheckpointInfo | None:
        for info in self.all:
            if info.id == id:
                return info
        return None

    def get_default(self) -> CheckpointInfo | None:
        # Go through the list of defaults first
        for id in self.default_ids:
            info = self.get(id)
            if info is not None:
                return info

        # Return the first model in the repository
        if len(self.all) > 0:
            return self.all[0]

        return None

    def get_closest(self, searchstr) -> Any | None:
        applicable = sorted([info for info in self.all if searchstr in info.id], key=lambda x: len(x.id))
        if len(applicable) > 0:
            return applicable[0]
        return None

    def get_ids(self) -> list:
        return sorted([x.id for x in self.all])

    def reload(self):
        self.all.clear()
        self.add(self.dirpath)

    def add(self, path):
        def add_file(path):
            confpath = path.with_suffix(".yaml")
            if not confpath.is_file():
                confpath = self.config

            self.all.append(CheckpointInfo(path, model_hash(path), confpath))

        def add_dir(dirpath):
            for e in self.extensions:
                for p in dirpath.glob(f"*.{e}"):
                    add_file(p)

        if path.exists():
            if path.is_dir():
                add_dir(path)
            elif path.is_file():
                add_file(path)
        else:
            printerr(f"Path does not exist: {path}")
