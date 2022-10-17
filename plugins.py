import os
import sys
import traceback
from pathlib import Path

import gradio

# Constants
INIT_NONE = 0
INIT_ENABLE = 1
INIT_RUN = 2


class Plugin:
    filename = None
    init_level = 0
    enabled = False

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        raise NotImplementedError()

    def id(self):
        return Path(self.filename).stem

    # The description method is currently unused.
    # To add a description that appears when hovering over the title, amend the "titles"
    # dict in script.js to include the script title (returned by title) as a key, and
    # your description as the value.
    def describe(self):
        return ""

    # Perform initialization on script enable or script run.
    # If the script is enabled on startup in the settings, we init right after the UI is launched.
    def init(self, level):
        pass

    def install(self):
        pass

    def uninstall(self):
        pass

    # Determines when the script should be shown in the dropdown menu via the returned value.
    # Return a list of pages to show on.
    # Valid values: ["text2img", "img2img", "extras", "settings"]
    # TODO Any other string will create a new tab by this name reserved for this script.
    def show(self, page: str):
        return True

    # How the script is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.
    def ui(self, page):
        pass

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.
    def run(self, *args):
        raise NotImplementedError()

    def img2img(self, img):
        pass

    def txt2img(self, img):
        pass

    def img2txt(self, img):
        pass

    def on_run_start(self):
        pass

    def on_postprocess_run_params(self, p):
        pass

    def on_batch_start(self, p):
        pass

    def on_postprocess_batch_params(self, p):
        pass

    def on_step_start(self, latent):
        pass

    def on_step_condfn(self):
        pass

    def on_step_end(self):
        pass

    def on_postprocess_batch(self, imgs):
        """
        Called at the end of each batch.
        Args:
            imgs: A list of the batch's image output.

        Returns:
        """
        pass

    def on_postprocess_image(self, img):
        """
        Called for every image output of each batch.
        Args:
            img: the output image to process, in cv2 rgb format. Use utilities to convert

        Returns: the modified image.
        """
        pass

    def on_postprocess_path(self, img, path: Path):
        """
        Post-process the save path for an output image.
        Args:
            img: The image in cv2 RGB format.
            path: The path that the image was saved to.

        Returns:
        """

    def on_image_saved(self, img, path: Path):
        """
        An ouput image has been saved.
        Args:
            img: The image in cv2 RGB format.
            path: The path that the image was saved to.

        Returns:
        """

    def on_batch_end(self):
        pass

    def on_run_end(self):
        pass

    def on_run_interrupted(self):
        pass

    # Allows accessing script's attributes by indexing e.g. script['on_run_start']
    def get(self, varname):
        return getattr(self, varname)


class ListDispatch:
    def __init__(self, ref, log=False):
        self.ref = ref
        self.log = log

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            ret = []
            if self.log:
                print(f"PluginDispatch({name}) to {len(self.ref)} scripts")

            for plugin in self.ref:
                if plugin.enabled:
                    v = getattr(plugin, name)(*args, **kwargs)
                    ret.append(v)
            return ret

        return wrapper


plugin_infos = []  # Plugin infos (script class, filepath)
plugins = []  # Loaded plugins
dispatch = ListDispatch(plugins, True)


def get(plugname):
    for plugin in plugins:
        if plugin.id() == plugname:
            return plugin

    return None

def read_plugin_infos(basedir):
    plugin_infos.clear()
    if not os.path.exists(basedir):
        return

    for filename in sorted(os.listdir(basedir)):
        path = os.path.join(basedir, filename)

        if not os.path.isfile(path):
            continue

        try:
            with open(path, "r", encoding="utf8") as file:
                text = file.read()

            from types import ModuleType
            compiled = compile(text, path, 'exec')
            module = ModuleType(filename)
            exec(compiled, module.__dict__)

            for key, script_class in module.__dict__.items():
                if type(script_class) == type and issubclass(script_class, Plugin):
                    plugin_infos.append((script_class, path))

        except Exception:
            print(f"Error loading script: {filename}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)


def plugin_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        res = func(*args, **kwargs)
        return res
    except Exception:
        print(f"Error calling: {filename}/{funcname}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    return default


def setup_ui(self):
    for plugclass, plugpath in plugin_infos:
        plug = plugclass()
        plug.filename = plugpath

        self.plugins.append(plug)

    # Gradio tab list, one tab per plugin
    # create plugin controls like this: controls = wrap_call(plug.ui, plug.filename, "ui")
    # and add them to the tab list like this: tabs.append(gradio.inputs.Tab(plug.title(), controls))
    tabs = []
    for plug in self.plugins:
        controls = plugin_call(plug.ui, plug.filename, "ui")
        tabs.append(gradio.inputs.Tab(plug.title(), controls))

    return tabs


def reload(basedir):
    plugins.clear()

    plugin_infos.clear()
    read_plugin_infos(basedir)

    print(f'Reloaded plugins, {len(plugin_infos)} found')


def reload_sources(self):
    for index, plug in list(enumerate(self.plugins)):
        with open(plug.filename, "r", encoding="utf8") as file:
            args_from = plug.args_from
            args_to = plug.args_to
            filename = plug.filename

            from types import ModuleType

            # Exec the plug code
            text = file.read()
            compiled = compile(text, filename, 'exec')
            module = ModuleType(plug.filename)
            exec(compiled, module.__dict__)

            # Instantiate plug classes
            for key, plugclass in module.__dict__.items():
                if type(plugclass) == type and issubclass(plugclass, Plugin):
                    self.plugins[index] = plugclass()
                    self.plugins[index].filename = filename
                    self.plugins[index].args_from = args_from
                    self.plugins[index].args_to = args_to
