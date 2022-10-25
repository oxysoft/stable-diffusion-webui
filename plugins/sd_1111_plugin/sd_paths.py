from core import paths

config = paths.repodir / 'stable_diffusion' / 'configs/stable-diffusion/v1-inference.yaml'
ckpt = "models/Stable-diffusion/sd-v1-4.ckpt"
ckpt_dir = paths.modeldir / 'Stable-diffusion'
embeddings_dir = paths.rootdir / 'embeddings'
hypernetwork_dir = ckpt_dir / 'hypernetworks'
vae_path = None

ckpt_dir = ckpt_dir.as_posix()  # TODO we should refactor all the code to use pathlib instead, it's 100x more readable
