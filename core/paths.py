import sys
from pathlib import Path


# script_path = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
rootdir = Path(__file__).resolve().parent.parent
modeldir = rootdir / "models"
repodir = rootdir / "plugin-repos"
plugindir = rootdir / "modules"
embeddingdir = rootdir / "embeddings"

sys.path.insert(0, rootdir.as_posix())

# search for directory of stable diffusion in following places
# path_dirs = [
#     (sd_path, 'ldm', 'Stable Diffusion', []),
#     (os.path.join(sd_path, '../taming-transformers'), 'taming', 'Taming Transformers', []),
#     (os.path.join(sd_path, '../CodeFormer'), 'inference_codeformer.py', 'CodeFormer', []),
#     (os.path.join(sd_path, '../BLIP'), 'models/blip.py', 'BLIP', []),
#     (os.path.join(sd_path, '../k-diffusion'), 'k_diffusion/SDSampler.py', 'k_diffusion', ["atstart"]),
# ]
#
# paths = {}
#
# for d, must_exist, what, options in path_dirs:
#     path = Path(script_path, d, must_exist).resolve()
#
#     if not path.exists():
#         print(f"Warning: {what} not found at path {path}", file=sys.stderr)
#     else:
#         d = os.path.abspath(d)
#         if "atstart" in options:
#             sys.path.insert(0, d)
#         else:
#             sys.path.append(d)
#
#         paths[what] = d

# Convert above loop to a function (


