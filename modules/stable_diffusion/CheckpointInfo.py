from pathlib import Path

class CheckpointInfo:
    def __init__(self, path, hash, config):
        self.path = path
        self.hash = hash

        self.configpath = config
        self.id = Path(path).relative_to(path.parent.parent).stem
        self.title = f'{self.id} [{hash}]'

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__[name]

    def __str__(self):
        return f'CheckpointInfo({self.id}, {self.title}, {self.path}, {self.hash}, {self.configpath})'

    def __repr__(self):
        return self.id