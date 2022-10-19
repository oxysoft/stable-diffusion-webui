import sys
import traceback
from pathlib import Path

# Constants
from core.jobs import JobParams

INIT_NONE = 0
INIT_ENABLE = 1
INIT_RUN = 2


class Plugin:
    id = ""
    filename = None
    init_level = 0
    enabled = False

    def __init__(self, filename):
        if filename is not None:
            self.filename = Path(filename)
            self.id = Path(self.filename).stem
        else:
            self.id = self.__class__.__name__

    def new_job(self, funcname, parameters: JobParams):
        """
        Return a new job
        """
        from core import jobs
        parameters.plugin = self
        return jobs.new_job(self.id, funcname, parameters)

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        raise NotImplementedError()

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

    def args(self, parser):
        pass

    # def img2img(self, img):
    #     pass
    #
    # def txt2img(self, img):
    #     pass
    #
    # def img2txt(self, img):
    #     pass

    def on_run_start(self):
        pass

    def on_postprocess_run_params(self, p):
        pass

    def on_batch_start(self, p):
        pass

    def on_postprocess_batch_params(self, p):
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
    """
    A class to automatically call a function on every plugin.
    """

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


plugins_info = []  # Plugin infos (script class, filepath)
plugins = []  # Loaded modules
dispatch = ListDispatch(plugins, True)


# def setup_ui(self):
#     for plugclass, plugpath in plugin_infos:
#         plug = plugclass()
#         plug.filename = plugpath
#
#         self.modules.append(plug)
#
#     # Gradio tab list, one tab per plugin
#     # create plugin controls like this: controls = wrap_call(plug.ui, plug.filename, "ui")
#     # and add them to the tab list like this: tabs.append(gradio.inputs.Tab(plug.title(), controls))
#     tabs = []
#     for plug in self.modules:
#         controls = plugin_call(plug.ui, plug.filename, "ui")
#         tabs.append(gradio.inputs.Tab(plug.title(), controls))
#
#     return tabs

def list():
    """
    Return a list of all plugins (string IDs only)
    """
    return [plug.id() for plug in plugins]  # TODO pack more info


def get(plugid):
    for plugin in plugins:
        if plugin.id() == plugid:
            return plugin

    return None


def info(plugid):
    plug = get(plugid)
    if plug:
        return plug.info()

    return None


def invoke(plugid, funcname, default=None, *args, **kwargs):
    """
    Invoke a plugin, may return a job object.
    """
    try:
        plug = get(plugid)
        if plug:
            return getattr(plug, funcname)(*args, **kwargs)
    except Exception:
        print(f"Error calling: {plugid}/{funcname}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    return default


def load_py(path):
    """
    Manually load a plugin at the given path
    """
    # import modules.StableDiffusion.StableDiffusionPlugin
    from types import ModuleType

    with open(path, "r", encoding="utf8") as file:
        filename = path

        # Exec the plug code

        sys.path.append(path.parent)
        sys.path.append(path.parent.parent)

        print(sys.path)
        text = file.read()
        compiled = compile(text, filename, 'exec')
        module = ModuleType(path.as_posix())
        exec(compiled, module.__dict__)

        # Instantiate plug classes
        for key, plugclass in module.__dict__.items():
            if type(plugclass) == type and issubclass(plugclass, Plugin):
                plug = plugclass()
                plug.filename = filename
                plugins.append(plug)


def load_dir(dirpath):
    if not dirpath.exists():
        return

    # Read the modules from the plugin directory
    for filename in dirpath.glob("*.py"):
        try:
            with open(filename, "r", encoding="utf8") as file:
                text = file.read()

            from types import ModuleType
            compiled = compile(text, filename, 'exec')
            module = ModuleType(filename.as_posix())
            exec(compiled, module.__dict__)

            for key, script_class in module.__dict__.items():
                if type(script_class) == type and issubclass(script_class, Plugin):
                    plugins_info.append((script_class, filename))

        except Exception:
            print(f"Error loading script: {filename}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    # Reload instances
    for plug in plugins_info:
        load_py(plug.filename)

    print(f'Reloaded modules, {len(plugins_info)} found')
