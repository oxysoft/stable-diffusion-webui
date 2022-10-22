# stable-core

# Mission

- **Server/Client Design:**  Clients can be UIs designed for this backend, or bridge to other apps like blender nodes, kdenlive clips, effects, etc. Currently using flask with flask-sockio since it's very fast to use.
- **Job Management:** Generate some data or transform some other data.  Currently it's a simple queue. In the future it could be scaled up to allow deferring to multiple backend nodes such as a cluster of GPUs, horde, etc.
- **Plugin Ecosystem:** Plugin is a wrapper around models, packages, techniques, features, etc. it handles all installation for its libraries and implements backend jobs. A CLI script wizard to instantly create a new plugin and start working on it. Acts a bit like a package manager for AI art, implement all your ideas and favorite models into stable-core to benefit from multiple GUIs and chain it with other community plugins, all designed for creative coding. Installation and repositories is all managed by each plugin, no need to think about this stuff anymore.
- **Cloud Deploy:** Instantly render on runpod, vast.ai in just a few clicks. Paste in your SSH information to copy your configuration and your installation will automatically be installed and your local jobs are deferred to the instance.
- **Multi-modal:** text, images, audio types as well. Each plugin job specifies the input and output so that we can transform the data around.

Each plugin clearly announces its functions and parameters, so one generic UI drawer code to render them all.
The in/out parameters allow to create node UI to chain plugin jobs, a list macro, scripting logic, etc.

## Contributions

I launch directly with `webui.sh` on linux which handles basic pip requirements as in AUTOMATIC1111, and this script is pretty much unchanged.
In Pycharm it also works to run `launch.py` for debugging, but I'm not sure if it works because I launched `webui.sh` first or it might be using my local installed packages instead of venv, not exactly sure but it works and I can debug.

I deleted the webui-user scripts since we won't be doing CLI arguments anymore, at least not in a way you would want to save them for configuration. There didn't seem to be anything else important for end users in the webui-user script but we may wanna review this, there seemed to be useful things for development purposes. We'll do a proper configuration file for the core which has basic things like the port for the server.

Contribution points for anyone who'd like to help.

- **Interactive Shell:** it would be cool to embed an interactive CLI interface into the server to use it without a UI, idk how to do this with flask though. (right now we're simply launching through app.run()) 
- **Plugins:** We already 'have' a bunch of plugins courtesy of AUTOMATIC1111 (mainly upscalers), the code still needs to be ported for each of them, however it's best to wait for the plugin and job API to solidify more as I'm working on the SD plugin. Then we can try to implement new ones. In the meantime...
   - We need a **Plugin Shell Script** (written in python) for the following features...
   - **Discovery:** Figure out how to host plugins on github and automatically collect them for listing. I'm pretty sure a bunch of other projects do it, it has to be possible somehow, maybe check with `https://vimawesome.com/` how they do it or if it's all manual.
   - **Creation:** Create a new plugin, ready to work on it and push to a repository. This is a directory with __init__ and a class extending `Plugin`, `stable_diffusion` is currently the best example we have. The directory will be used as its identifier for client/server communication so it should be all lowercase, and a valid module name so no dashes.
   - **Update:** Update an existing plugin with git pull on it.
- **UI:** we don't have a UI yet, I will write one in Dear ImGUI as soon as SD plugin is usable.
- **Authentication:** session system to connect with a passwords, ssh, etc. no sharing without this obviously.

### Coding Standards

- **KISS:** We abid KISS, must be able to read and understood whole thing in under an hour. Always consider more than one approach, pick the simplest. Adding more code to a function is always fine, but adding additional class, function, or even fields/properties introduces top-level complexity and should always lead to some API reflection.
- **Documentation:** There is a severe lack of quality documentation in the world of programming, see `launch.py` for a good demo of proper documentation. Long methods are fine, but add big header comments with titles. Any code where it isn't immediately obvious _why we do it_ or _why that way exactly_, should be commented.
- **Stability:** Don't use exceptions for simple stuff. Fail gracefully with an error message, recognize that the error state is possible as part of the API, and return a default value or ability to handle the error. The core must be able to remain up 24/7 for weeks at a time. We should try to keep the backend core running when maxing out VRAM, either by preventing or by running each plugin on separate web-core process so the backend can keep running even if a plugin results in OOM.  
- **Orthogonality:** Huge emphasis on local responsabilities, e.g. don't do any saving or logging directly as part of a job. Instead, report the progress and output data, and let the specifics be handled externally by some other specialized component. Avoid passing some huge bags of options, and if there's a lot of default values put them in a nested option class for the related job, or architecture the code such as to be able to post-process the values and apply defaults inside the plugin itself.
- **Unit Testing:** not planned currently but test suites could certainly be useful eventually, especially on individual plugins that might change a lot like StableDiffusionPlugin.

**Formatting:**
- 4 spaces indent
- Prefer Pathlib Path over filename strings

The rest is all good, 

## Roadmap:
1. ~Core backend components (server, jobs, plugins) to a usable state.~
2. Run the StableDiffusionPlugin txt2img job from CLI
3. Write a UI to run the job in and see progress.
4. Port some upscalers so we can see the job workflow in action.

## Plugin

Let me know if any other idea comes to mind

* **AUTO1111 StableDiffusion:** txt2img, img2img
* **Diffusers StableDiffusion:** txt2img, img2img
* **VQGAN+CLIP / PyTTI:** txt2img, img2img
* **DiscoDiffusion:** txt2img, img2img
* **CLIP Interrogate:** img2txt
* **Dreambooth**: data2ckpt
* **StyleGAN:** data2ckpt, img2img
* **2D Transforms:** simple 2D transforms like translate, rotate, and scale.
* **3D Transforms:** 3D transforms using virtual depth like rotating a sphere OR predicted depth from AdaBins+MiDaS. Could implement depth guidance to try and keep the depth more stable.
* **Guidance:** these plugins guide the generation plugins.
   * **CLIP Guidance:** guidance using CLIP models.
   * **Lpips Guidance:** guidance using lpips
   * **Convolution Guidance:** guidance using convolutions. (edge_weight in PyTTI)
* **Audio Analysis:** img2num, turn audio inputs into numbers for audio-reactivity, using FFT and stuff like that. Can maybe use Magenta.
* **Palette Match:** img2img, adjust an image's palette to match an input image.
* **Flow Warp:** img2img, displace an image using estimated flow between 2 input images.
* **Prompt Wildcards:** txt2txt
* **Whisper:** audio2txt
* Upscalers:
  * **RealSR:** img2img, on Linux this is easily installed thru AUR with `realsr-ncnn-vulkan`
  * **BasicSR:** img2img, port
  * **LDSR:** img2img
  * **CodeFormer:** img2img, port
  * **GFPGAN:** img2img, port
* **MetaPlugin:** a plugin to string other plugins together, either with job macros or straight-up python. Could be done without a plugin but this allows all clients to automatically support these features.

## Progress Report - 10/20

- Server/Client design: ready. (really the the minimum)
- Plugins: ready. See the contribution section above to see what's left
- SD plugin: 75%, hypernetworks and textinv in refactoring.
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
