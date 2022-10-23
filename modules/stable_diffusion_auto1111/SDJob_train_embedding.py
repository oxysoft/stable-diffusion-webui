from pathlib import Path

from core.jobs import JobParams


class SDJob_train_embedding(JobParams):
    def get_plugin_impl(self):
        return 'stable_diffusion', 'data2ckpt_embedding'

    def __init__(self,
                 name:str,
                 datadir:Path,
                 model,
                 lr:float=0.00001,  # TODO idk what the default should be
                 num_repeats:int=1,
                 w:int=512,
                 h:int=512,
                 steps:int=500,  # TODO idk what the default should be
                 save_every:int=100,  # TODO idk what the default should be
                 template_file:Path=None):  # TODO no clue what this is
        super().__init__()
        self.name = name
        self.datadir = datadir
        self.model = model
        self.learn_rate = lr
        self.training_width = w
        self.training_height = h
        self.steps = steps
        self.num_repeats = num_repeats
        self.save_embedding_every = save_every
        self.template_file = template_file