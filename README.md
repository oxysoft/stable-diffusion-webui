# stable-core

# Mission

- **Server/Client Design:**  Clients can be UIs designed for this backend, or bridge to other apps like blender nodes, kdenlive clips, effects, etc. Currently using flask with flask-sockio since it's very fast to use.
- **Job Management:** Generate some data or transform some other data.  Currently it's a simple queue. In the future it could be scaled up to allow deferring to multiple backend nodes such as a cluster of GPUs, horde, etc.
- **Plugin Ecosystem:** Plugin is a wrapper around models, packages, techniques, features, etc. it handles all installation for its libraries and implements backend jobs. A CLI script wizard to instantly create a new plugin and start working on it. Acts a bit like a package manager for AI art, implement all your ideas and favorite models into stable-core to benefit from multiple GUIs and chain it with other community plugins, all designed for creative coding. Installation and repositories is all managed by each plugin, no need to think about this stuff anymore.
- **Cloud Deploy:** Instantly render on runpod, vast.ai in just a few clicks. Paste in your SSH information to copy your configuration and your installation will automatically be installed and your local jobs are deferred to the instance.
- **Multi-modal:** text, images, audio types as well. Each plugin job specifies the input and output so that we can transform the data around.

Each plugin clearly announces its functions and parameters, so one generic UI drawer code to render them all.
The in/out parameters allow to create node UI to chain plugin jobs, a list macro, scripting logic, etc.

## Contributions:

Contribution points for anyone who'd like to help.

- **Interactive Shell:** it would be cool to embed an interactive CLI interface into the server to use it without a UI, idk how to do this with flask though. (just using app.run() to launch it) 
- **Plugins:** We already 'have' a bunch of plugins courtesy of AUTOMATIC1111, mainly upscalers. The code still needs to be ported for each of them. Then after that we can try to implement new ones.
- **UI:** we don't have a UI yet, I will write one in Dear ImGUI as soon as SD plugin is usable.
- **Authentication:** session system to connect with a passwords, ssh, etc. no sharing without this obviously.
- **Plugin Shell Script:**
   - We need a CLI script to interact with plugins. (written in Python)
   - Discoverery: Figure out how to host plugins on github and automatically collect them for listing.
   - Creation: Create a new plugin, ready to work on it and push to a repository.
   - Update: Update an existing plugin with git pull.

### Coding Standards

- **KISS:** we abid KISS, must be able to read and understood whole thing in under an hour. Always consider more than one approach, pick the simplest. As few moving parts as possible.
- **Robust:** Avoid crashing as much as possible, we should try to keep the backend core running when maxing out VRAM. We can maybe run plugins on a separate process so the backend can keep running even if a plugin results in OOM.
- **Orthogonal:** Avoid global states as much as possible, emphasis on locality.
- **Unit Testing:** not planned for the first releases but a test suite could be useful.

### Formatting
- 4 spaces indent
- Prefer Pathlib Path over filename strings


## Roadmap:
1. Core backend components (server, jobs, plugins)
2. Run the StableDiffusionPlugin txt2img job from CLI
3. Write a UI to run the job in and see progress.
4. Port some upscalers so we can see the job workflow in action.

## Plugin

Everything that comes to mind

* StableDiffusion: txt2img, img2img
* VQGAN+CLIP / PyTTI: txt2img, img2img
* DiscoDiffusion: txt2img, img2img
* CLIP Interrogate: img2txt
* 2D Transforms: simple 2D transforms like translate, rotate, and scale.
* 3D Transforms: 3D transforms using virtual depth like rotating a sphere OR predicted depth from AdaBins+MiDaS. Could implement depth guidance to try and keep the depth more stable.
* Guidance: these plugins guide the generation plugins.
   * CLIP Guidance: guidance using CLIP models.
   * Lpips Guidance: guidance using lpips
   * Convolution Guidance: guidance using convolutions
* Audio Analysis: img2num, turn audio inputs into numbers for audio-reactivity, using FFT and stuff like that. Can maybe use Magenta.
* Palette Match: img2img, adjust an image's palette to match an input image.
* Flow Warp: img2img, displace an image using estimated flow between 2 input images.
* Prompt Wildcards: txt2txt
* Whisper: audio2txt
* Upscalers:
  * RealSR: img2img, on Linux this is easily installed thru AUR with `realsr-ncnn-vulkan`
  * BasicSR: img2img, port
  * LDSR: img2img
  * CodeFormer: img2img, port
  * GFPGAN: img2img, port
* Metaplugin: a plugin to string other plugins together, either with job macros or straight-up python. Could be done without a plugin but this allows all clients to automatically support these features.

## Progress Report - 10/20

- Server/Client design: ready.
- Plugins: loading and API ready. See the contribution section above to see what's left.
  - Configuration: missing a good approach, CLI is evil and the previous approach for options also evil.
- SD plugin: almost functional, hypernetworks and textinv in refactoring.
- UI: starting as soon as SD plugin is done.

## Progress Report - 10/19

The server now boots up and we can import the StableDiffusion plugin, and even instantiate it without crashing.
The SD plugin processes are being refactored into the job system as JobParameters, which we can extend.
The ProcessResult had too many values being copied around. Instead we are now keeping them in the JobParameters object. 

So the plugin announces its job signatures like this: `name, function, input type, output type, parameter class`
Each invocation function returns one or multiple jobs, and each job has an associated param object to configure it.

A lot of useless UI shit mixed into the backend, we're mostly restarting from scratch for the gradio UI.

AUTOMATIC1111 is still not responding and I don't know any other way to contact him so don't know if we have him on-board. The project must be renamed to stable-core or something not stable-diffusion related.

## Progress Report - 10/18

Current state of things if you wish to contribute and speed things up:

- Many modules have been moved to plugins, they must be reviewed one by one and adapted into its Plugin class
- Exorcise all reference of `shared`, CLI args, and options.
- **Must figure out a real backend solution, not this gradio stuff**
- We will probably rewrite the UI completely, old pieces can be adapted if necessary. Since we can move each plugin's UI into its own plugin file the UI will be a lot easier to improve in the future.
- There are more modules remaining, some are just utility functions
