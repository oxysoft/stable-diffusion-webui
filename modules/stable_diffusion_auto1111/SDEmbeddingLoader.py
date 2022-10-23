import traceback
from pathlib import Path

import torch
from PIL import Image
from torch.nn import Embedding

from core import devicelib
from core.printing import printerr
from modules.stable_diffusion_auto1111.TextInv64 import embedding_from_b64, extract_image_data_embed


class SDEmbeddingLoader:
    def __init__(self, directory:Path):
        self.id_lookup = {}
        self.embeddings = {}
        self.directory = directory
        self.directory_mtime = None

    def reload(self, model):
        mt = self.directory.stat().st_mtime
        if self.directory_mtime is not None and mt <= self.directory_mtime:
            return
        self.directory_mtime = mt

        self.id_lookup.clear()
        self.embeddings.clear()

        for file in self.directory.iterdir():
            try:
                if file.stat().st_size == 0:
                    continue
                self.load_file(file)
            except Exception:
                printerr(f"Error loading emedding {file}:")
                printerr(traceback.format_exc())
                continue

        print(f"Loaded {len(self.embeddings)} textual inversion embeddings.")
        print("Embeddings:", ', '.join(self.embeddings.keys()))

    def load_file(self, path, model):
        name = path.stem

        if path.suffix.upper() in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            embed_image = Image.open(path)
            if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                name = data.get('name', name)
            else:
                data = extract_image_data_embed(embed_image)
                name = data.get('name', name)
        else:
            data = torch.load(path, map_location="cpu")

        # Textual Inversion embeddings
        if 'string_to_param' in data:
            param_dict = data['string_to_param']
            if hasattr(param_dict, '_parameters'):
                param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1]

        # Diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
            assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise Exception(f"Couldn't identify {name} as neither textual inversion embedding nor diffuser concept.")

        vec = emb.detach().to(devicelib.device, dtype=torch.float32)
        embedding = Embedding(vec, name)
        embedding.step = data.get('step', None)
        embedding.sd_checkpoint = data.get('hash', None)
        embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)

        self.add_embedding(embedding, model)

    def add_embedding(self, embedding, model):
        self.embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenizer([embedding.name], add_special_tokens=False)['input_ids'][0]

        first_id = ids[0]
        if first_id not in self.id_lookup:
            self.id_lookup[first_id] = []

        self.id_lookup[first_id] = sorted(self.id_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.id_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None