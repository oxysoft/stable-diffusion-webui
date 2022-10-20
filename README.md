# stable-core

# Mission

The long-term goal is to make a backend like this:

- **Server/Client Architecture:**  Clients can be UIs designed for this backend, or bridge to other apps like blender nodes, kdenlive clips, effects, etc. Currently using flask with flask-sockio since it's very fast to use.
- **Job Management:** Generate some data or transform some other data.  Currently it's a simple queue. In the future it could be scaled up to allow deferring to multiple backend nodes such as a cluster of GPUs, horde, etc.
- **Plugin Ecosystem:** Wrapper around models, packages, techniques, features, etc. handles all installation and implements backend jobs for them. A CLI script wizard to instantly create a new plugin and start working on it. Acts a bit like a package manager for AI art, implement all your ideas and favorite models into stable-core to benefit from multiple GUIs and chain it with other community plugins, all designed for creative coding. Installation and repositories is all managed by each plugin, no need to think about this stuff anymore.
- **Instant Cloud Deploy:** runpod, vast.ai in just a few clicks. Paste in your SSH information to copy your configuration and your installation will automatically be installed and your local jobs are deferred to the instance.
- **Multi-modal:** text, images, audio types as well. Each plugin job specifies the input and output so that we can transform the data around.

It's built on the tried and true codebase by AUTOMATIC1111. The backend abids KISS, whole thing can be read and understood in under in an hour.

UIs can be written as clients, I will do DearImGUI, but gradio would be cool as well for colab. 
Each plugin clearly announces its functions and parameters, so one generic UI drawer code to render them all.
The in/out parameters allow to create node UI to chain plugin jobs, a list macro, scripting logic, etc.

**Coding Standards:**
- 4 spaces indent
- Prefer Pathlib Path over filename strings


## Core/Plugin Refactor Progress - 10/19

The server now boots up and we can import the StableDiffusion plugin, and even instantiate it without crashing.
The SD plugin processes are being refactored into the job system as JobParameters, which we can extend.
The ProcessResult had too many values being copied around. Instead we are now keeping them in the JobParameters object. 

So the plugin announces its job signatures like this: `name, function, input type, output type, parameter class`
Each invocation function returns one or multiple jobs, and each job has an associated param object to configure it.

A lot of useless UI shit mixed into the backend, we're mostly restarting from scartch for the gradio UI.

Contribution points: 

- Obviously I am trying to get the SD plugin working first with img2img and txt2img jobs, then all the upscalers are mostly the same. 
- It would be cool to embed a CLI interface into the server but idk how to do this with flask, I'm using app.run(). 
- Missing a UI and the Stable Diffusion plugin is in shambles because still refactoring. A lot of the API points are missing for a good UI
- Need to figure out sessions with connection methods like passwords, ssh, etc. otherwise anyone can request jobs if u share (lol)
- Need to figure out how we can get an efficient system where plugins are hosted on github and collect them for listing.
- Removing a lot of cli args and options and using job params where possible

AUTOMATIC1111 is still not responding and I don't know any other way to contact him so don't know if we have him on-board. The project must be renamed to stable-core or something not stable-diffusion related.

## Core/Plugin Refactor Progress - 10/18

If you wish to contribute and speed things up, this is the current state of things:

- Many modules have been moved to plugins, they must be reviewed one by one and adapted into its Plugin class
- StableDiffusionPlugin is complex and broken up into several files.
   - The processing stuff can probably stay.
   - Sort out what is going on with the 'hijack' and 'hypernetwork' things, streamline that stuff
- We must exorcise the calls to `shared` across every plugin
- **Must figure out a real backend solution, not this gradio stuff**
- **Idk yet if the options stuff is compatible with the plugin architecture and how much needs refactoring**. I think it looks good and we can ask plugins to return an options_section(), but need to verify.
- We will probably rewrite the UI completely, old pieces can be adapted if necessary. Since we can move each plugin's UI into its on plugin file the UI will be a lot easier to improve in the future.
- There are more modules, some are just utility functions

## Features
[Detailed feature showcase with images](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features):
- Original txt2img and img2img modes
- One click install and run script (but you still must install python and git)
- Outpainting
- Inpainting
- Prompt Matrix
- Stable Diffusion Upscale
- Attention, specify parts of text that the model should pay more attention to
    - a man in a ((tuxedo)) - will pay more attention to tuxedo
    - a man in a (tuxedo:1.21) - alternative syntax
    - select text and press ctrl+up or ctrl+down to automatically adjust attention to selected text (code contributed by anonymous user)
- Loopback, run img2img processing multiple times
- X/Y plot, a way to draw a 2 dimensional plot of images with different parameters
- Textual Inversion
    - have as many embeddings as you want and use any names you like for them
    - use multiple embeddings with different numbers of vectors per token
    - works with half precision floating point numbers
- Extras tab with:
    - GFPGAN, neural network that fixes faces
    - CodeFormer, face restoration tool as an alternative to GFPGAN
    - RealESRGAN, neural network upscaler
    - ESRGAN, neural network upscaler with a lot of third party models
    - SwinIR and Swin2SR([see here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/2092)), neural network upscalers
    - LDSR, Latent diffusion super resolution upscaling
- Resizing aspect ratio options
- Sampling method selection
    - Adjust sampler eta values (noise multiplier)
    - More advanced noise setting options
- Interrupt processing at any time
- 4GB video card support (also reports of 2GB working)
- Correct seeds for batches
- Prompt length validation
     - get length of prompt in tokens as you type
     - get a warning after generation if some text was truncated
- Generation parameters
     - parameters you used to generate images are saved with that image
     - in PNG chunks for PNG, in EXIF for JPEG
     - can drag the image to PNG info tab to restore generation parameters and automatically copy them into UI
     - can be disabled in settings
- Settings page
- Running arbitrary python code from UI (must run with --allow-code to enable)
- Mouseover hints for most UI elements
- Possible to change defaults/mix/max/step values for UI elements via text config
- Random artist button
- Tiling support, a checkbox to create images that can be tiled like textures
- Progress bar and live image generation preview
- Negative prompt, an extra text field that allows you to list what you don't want to see in generated image
- Styles, a way to save part of prompt and easily apply them via dropdown later
- Variations, a way to generate same image but with tiny differences
- Seed resizing, a way to generate same image but at slightly different resolution
- CLIP interrogator, a button that tries to guess prompt from an image
- Prompt Editing, a way to change prompt mid-generation, say to start making a watermelon and switch to anime girl midway
- Batch Processing, process a group of files using img2img
- Img2img Alternative
- Highres Fix, a convenience option to produce high resolution pictures in one click without usual distortions
- Reloading checkpoints on the fly
- Checkpoint Merger, a tab that allows you to merge two checkpoints into one
- [Custom scripts](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts) with many extensions from community
- [Composable-Diffusion](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/), a way to use multiple prompts at once
     - separate prompts using uppercase `AND`
     - also supports weights for prompts: `a cat :1.2 AND a dog AND a penguin :2.2`
- No token limit for prompts (original stable diffusion lets you use up to 75 tokens)
- DeepDanbooru integration, creates danbooru style tags for anime prompts (add --deepdanbooru to commandline args)
- [xformers](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers), major speed increase for select cards: (add --xformers to commandline args)

## Installation and Running
Make sure the required [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) are met and follow the instructions available for both [NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (recommended) and [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs) GPUs.

Alternatively, use Google Colab:

- [Colab, maintained by Akaibu](https://colab.research.google.com/drive/1kw3egmSn-KgWsikYvOMjJkVDsPLjEMzl)
- [Colab, original by me, outdated](https://colab.research.google.com/drive/1Iy-xW9t1-OQWhb0hNxueGij8phCyluOh).

### Automatic Installation on Windows
1. Install [Python 3.10.6](https://www.python.org/downloads/windows/), checking "Add Python to PATH"
2. Install [git](https://git-scm.com/download/win).
3. Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4. Place `model.ckpt` in the `models` directory (see [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) for where to get it).
5. _*(Optional)*_ Place `GFPGANv1.4.pth` in the base directory, alongside `webui.py` (see [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) for where to get it).
6. Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

### Automatic Installation on Linux
1. Install the dependencies:
```bash
# Debian-based:
sudo apt install wget git python3 python3-venv
# Red Hat-based:
sudo dnf install wget git python3
# Arch-based:
sudo pacman -S wget git python3
```
2. To install in `/home/$(whoami)/stable-diffusion-webui/`, run:
```bash
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
```

### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing
Here's how to add code to this repo: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation
The documentation was moved from this README over to the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits
- Stable Diffusion - https://github.com/CompVis/stable-diffusion, https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- CodeFormer - https://github.com/sczhou/CodeFormer
- ESRGAN - https://github.com/xinntao/ESRGAN
- SwinIR - https://github.com/JingyunLiang/SwinIR
- Swin2SR - https://github.com/mv-lab/swin2sr
- LDSR - https://github.com/Hafiidz/latent-diffusion
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Doggettx - Cross Attention layer optimization - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- InvokeAI, lstein - Cross Attention layer optimization - https://github.com/invoke-ai/InvokeAI (originally http://github.com/lstein/stable-diffusion)
- Rinon Gal - Textual Inversion - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Idea for Composable Diffusion - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- xformers - https://github.com/facebookresearch/xformers
- DeepDanbooru - interrogator for anime diffusers https://github.com/KichangKim/DeepDanbooru
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)
