import html

import gradio as gr

from core import plugins
import shared


def create_embedding(name, initialization_text, nvpt):
    filename = plugins.StableDiffusion.textual_inversion.textual_inversion.create_embedding(name, nvpt, init_text=initialization_text)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def preprocess(*args):
    plugins.StableDiffusion.textual_inversion.preprocess.preprocess(*args)

    return "Preprocessing finished.", ""


def train_embedding(*args):

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram.py not possible'

    try:
        sd_hijack.undo_optimizations()

        embedding, filename = plugins.StableDiffusion.textual_inversion.textual_inversion.train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()

