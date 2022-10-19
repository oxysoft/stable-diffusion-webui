from pathlib import Path

class CheckpointInfo:
    def __init__(self, path, hash, config):
        self.path = path
        self.hash = hash

        self.configpath = config
        self.id = Path(path).relative_to(path.parent.parent).as_posix()
        self.title = f'{self.id} [{hash}]'