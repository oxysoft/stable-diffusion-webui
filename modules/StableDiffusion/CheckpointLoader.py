from core import paths
from core.modellib import model_hash
from core.printing import printerr
from modules.StableDiffusion.CheckpointInfo import CheckpointInfo


class CheckpointLoader:
    def __init__(self, subdirname, config, default_id=None):
        self.all = []
        self.config = config
        self.dirpath = paths.modeldir / subdirname
        self.default_model = default_id or "model"

        self.reload()

    def get_default(self):
        return self.get(self.default_model)

    def checkpoint_ids(self):
        return sorted([x.id for x in self.all])

    def get_closet_checkpoint_match(self, searchstr):
        applicable = sorted([info for info in self.all if searchstr in info.id], key=lambda x: len(x.id))
        if len(applicable) > 0:
            return applicable[0]
        return None

    def reload(self):
        self.all.clear()
        self.add_dir(self.dirpath)

    def add(self, path):
        if path.exists():
            if path.is_dir():
                self.add_dir(path)
            elif path.is_file():
                self.add_file(path)
        else:
            printerr(f"Path does not exist: {path}")

    def add_dir(self, dirpath):
        for path in dirpath.glob("*.ckpt"):
            self.add_file(path)
        for path in dirpath.glob("*.pt"):
            self.add_file(path)

    def add_file(self, path):
        config = path.with_suffix(".yaml")
        if not config.is_file():
            config = self.config

        self.all.append(CheckpointInfo(path, model_hash(path), config))

    def get(self, id):
        for info in self.all:
            if info.id == id:
                return info
        return None