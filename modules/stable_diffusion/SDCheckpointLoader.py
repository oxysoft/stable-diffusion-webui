import math

import bunch
import torch
from omegaconf import OmegaConf

from core import devicelib
from core.modellib import send_everything_to_cpu

import CheckpointInfo
from CheckpointLoader import CheckpointLoader

from ldm.util import instantiate_from_config, ismap
from core import plugins, devicelib, promptlib

from core.cmdargs import cargs
from core.devicelib import device
from core.options import opts
from optimizations import setup_for_low_vram


def get_state_dict_from_checkpoint(pl):
    if "state_dict" in pl:
        return pl["state_dict"]

    return pl

def flatten(elem):
    flattened = [flatten(children) for children in elem.children()]
    res = [elem]
    for c in flattened:
        res += c
    return res

def get_target_prompt_token_count(token_count):
    return math.ceil(max(token_count, 1) / 75) * 75


def add_circular_option_to_conv_2d():
    # TODO this is unused
    conv2d_constructor = torch.nn.Conv2d.__init__

    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode='circular', **kwargs)

    torch.nn.Conv2d.__init__ = conv2d_constructor_circular


class SDCheckpointLoader(CheckpointLoader):
    def __init__(self, plugin, *kargs):
        super().__init__(*kargs)
        self.plugin = plugin
        # self.fixes = None
        # self.comments = []

    def opt(self):
        return self.plugin.opt

    # TODO we must do the right thing
    # embedding_db = plugins.stable_diffusion.textual_inversion.textual_inversion.EmbeddingDatabase(cargs.embeddings_dir)

    def tokenize(self, text):
        _, remade_batch_tokens, _, _, _, token_count = self.clip.process_text([text])
        return remade_batch_tokens[0], token_count, get_target_prompt_token_count(token_count)

    def load(self, info: CheckpointInfo):
        if info is None:
            raise ValueError("Cannot load a null CheckpointInfo.")

        print(f"Loading SD from: {info.configpath}")
        model = instantiate_from_config(OmegaConf.load(info.configpath).model)
        self.load_into(model, info)

        # Low VRAM opt
        if self.plugin.opt.lowvram or self.plugin.opt.medvram: # we should call this what it actually is
            setup_for_low_vram(model, self.plugin.opt.medvram)
        else:
            model.to(devicelib.device)

        self.hijack(model)
        model.eval()
        model.info = info

        print(f"Model loaded.")
        return model

    def load_into_standalone(self, model, ickpt=None):
        if self.plugin.opt.lowvram or self.plugin.opt.medvram: # we should call this what it actually is
            send_everything_to_cpu()
        else:
            model.to(devicelib.cpu)

        self.undo_hijack(model)
        self.load_into(model, ickpt)
        self.hijack(model)

        if not self.plugin.opt.lowvram and not self.plugin.opt.medvram:
            model.to(devicelib.device)

        print(f"Weights loaded.")
        return model

    def load_into(self, model, info):
        """
        Load weights from a CheckpointInfo into the instantiated model.
        """
        opt = self.opt()

        print(f"Loading weights [{info.hash}] from {info.path}")
        pl = torch.load(info.path, map_location="cpu")

        if "global_step" in pl:
            print(f"Global Step: {pl['global_step']}")

        model.load_state_dict(get_state_dict_from_checkpoint(pl), strict=False)
        model.info = info
        model.state = bunch.Bunch()
        model.state.layers = flatten(model)

        # Memory Optimizations
        if opt.opt_channelslast:
            model.to(memory_format=torch.channels_last)
        if not opt.no_half:
            model.half()

        # TODO why is this here, what does this have to do with the model
        devicelib.dtype = torch.float32 if opt.no_half else torch.float16
        devicelib.dtype_vae = torch.float32 if opt.no_half or opt.no_half_vae else torch.float16

        # VAE loading ...................................
        vaepath = opt.vae_override or info.path.with_suffix(".vae.pt")
        if vaepath.exists():
            print(f"Loading VAE weights from: {vaepath}")
            ckpt = torch.load(vaepath, map_location="cpu")
            state = {k: v for k, v in ckpt["state_dict"].items() if k[0:4] != "loss"}
            model.first_stage_model.load_state_dict(state)

        model.first_stage_model.to(devicelib.dtype_vae)

    def set_circular(self, model, enable):
        if model.state.get('circular_enabled') == enable:
            return

        model.state.circular_enabled = enable

        for layer in [layer for layer in model.state.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def hijack(self, model):
        # Better embeddings... no clue what this does besides adding []() weight syntax
        model_embeddings = model.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)

        # Better CLIP... also dk what tis do
        clip = FrozenCLIPEmbedderWithCustomWords(model.cond_stage_model, self)

        model.cond_stage_model = clip
        model.state.clip = clip

    def undo_hijack(self, model):
        if type(model.cond_stage_model) == FrozenCLIPEmbedderWithCustomWords:
            model.cond_stage_model = model.cond_stage_model.wrapped

        model_embeddings = model.cond_stage_model.transformer.text_model.embeddings
        if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
            model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped

class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    """
    oxy: I think this implements text inversions and other stuff
    """
    def __init__(self, wrapped, hijack):
        super().__init__()
        self.wrapped = wrapped
        self.tokenizer = wrapped.tokenizer
        self.token_mults = {}
        self.comma_token = [v for k, v in self.tokenizer.get_vocab().items() if k == ',</w>'][0]

        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def process_text(self, texts):
        used_custom_terms = []
        remade_batch_tokens = []
        hijack_comments = []
        hijack_fixes = []
        token_count = 0

        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, fixes, multipliers = cache[line]
            else:
                remade_tokens, fixes, multipliers, current_token_count = self.tokenize_line(line, used_custom_terms, hijack_comments)
                token_count = max(current_token_count, token_count)

                cache[line] = (remade_tokens, fixes, multipliers)

            remade_batch_tokens.append(remade_tokens)
            hijack_fixes.append(fixes)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    def tokenize_line(self, line, used_custom_terms, hijack_comments):
        id_end = self.wrapped.tokenizer.eos_token_id

        if opts.enable_emphasis:
            parsed = promptlib.parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.wrapped.tokenizer([text for text, _ in parsed], truncation=False, add_special_tokens=False)["input_ids"]

        fixes = []
        remade_tokens = []
        multipliers = []
        last_comma = -1

        # wtf is going on here, need comments
        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, i)

                if token == self.comma_token:
                    last_comma = len(remade_tokens)
                elif opts.comma_padding_backtrack != 0 and max(len(remade_tokens), 1) % 75 == 0 and last_comma != -1 and len(remade_tokens) - last_comma <= opts.comma_padding_backtrack:
                    last_comma += 1
                    reloc_tokens = remade_tokens[last_comma:]
                    reloc_mults = multipliers[last_comma:]

                    remade_tokens = remade_tokens[:last_comma]
                    length = len(remade_tokens)

                    rem = int(math.ceil(length / 75)) * 75 - length
                    remade_tokens += [id_end] * rem + reloc_tokens
                    multipliers = multipliers[:last_comma] + [1.0] * rem + reloc_mults

                if embedding is None:
                    remade_tokens.append(token)
                    multipliers.append(weight)
                    i += 1
                else:
                    emb_len = int(embedding.vec.shape[0])
                    iteration = len(remade_tokens) // 75
                    if (len(remade_tokens) + emb_len) // 75 != iteration:
                        rem = (75 * (iteration + 1) - len(remade_tokens))
                        remade_tokens += [id_end] * rem
                        multipliers += [1.0] * rem
                        iteration += 1
                    fixes.append((iteration, (len(remade_tokens) % 75, embedding)))
                    remade_tokens += [0] * emb_len
                    multipliers += [weight] * emb_len
                    used_custom_terms.enqueue((embedding.name, embedding.checksum()))
                    i += embedding_length_in_tokens

        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)

        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add

        return remade_tokens, fixes, multipliers, token_count

    def forward(self, text):
        batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.process_text(text)

        self.hijack.comments += hijack_comments

        if len(used_custom_terms) > 0:
            self.hijack.comments.append("Used embeddings: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            self.hijack.fixes = []
            for unfiltered in hijack_fixes:
                fixes = []
                for fix in unfiltered:
                    if fix[0] == i:
                        fixes.append(fix[1])
                self.hijack.fixes.append(fixes)

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                    multipliers.append(batch_multipliers[j][:75])
                else:
                    tokens.append([self.wrapped.tokenizer.eos_token_id] * 75)
                    multipliers.append([1.0] * 75)

            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), axis=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1

        return z

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        remade_batch_tokens = [[self.wrapped.tokenizer.bos_token_id] + x[:75] + [self.wrapped.tokenizer.eos_token_id] for x in remade_batch_tokens]
        batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]

        tokens = torch.asarray(remade_batch_tokens).to(device)
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)

        if opts.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(batch_multipliers_of_same_length).to(device)
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z


class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = embedding.vec
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            vecs.append(tensor)

        return torch.stack(vecs)


