import importlib
import sys
import traceback
from pathlib import Path

# Constants
from core.jobs import JobParams
from core.printing import printerr

INIT_NONE = 0
INIT_ENABLE = 1
INIT_RUN = 2


class Plugin:
    def __init__(self, dirpath: Path, id: str = None):
        if dirpath is not None and id is None:
            self.dir = Path(dirpath)
            self.id = id or Path(self.dir).stem
        elif id is not None:
            self.dir = None
            self.id = id
        else:
            self.dir = None
            self.id = self.__class__.__name__

    def new_job(self, name, jobparams: JobParams):
        """
        Return a new job
        """
        from core import jobs
        jobparams.plugin = self
        return jobs.new_job(self.id, name, jobparams)

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

    def jobs(self):
        """Announce the jobs we can handle"""
        return dict()

    def install(self):
        pass

    def uninstall(self):
        pass

    # Allows accessing script's attributes by indexing e.g. script['on_run_start']
    def get(self, varname):
        return getattr(self, varname)


plugin_dirs = []  # Plugin infos (script class, filepath)
plugins = []  # Loaded modules


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

def list_ids():
    """
    Return a list of all plugins (string IDs only)
    """
    return [plug.id for plug in plugins]


def get(plugid):
    """
    Get a plugin instance by ID.
    """
    if isinstance(plugid, Plugin):
        return plugid

    for plugin in plugins:
        if plugin.id == plugid:
            return plugin

    return None


def info(plugid):
    """
    Get a plugin's info by ID
    """
    plug = get(plugid)
    if plug:
        return dict(id=plug.id,
                    jobs=plug.jobs(),
                    title=plug.title(),
                    description=plug.describe())

    return None


def load(path: Path):
    """
    Manually load a plugin at the given path
    """
    import inspect
    from types import ModuleType

    # Find classes that extend Plugin in the module
    try:
        sys.path.append(path.as_posix())
        plugin_dirs.append(path)

        # # this is missing a bunch of module files for some reason...
        # mod = importlib.import_module(f'modules.{path.stem}')
        #
        # for name, obj in inspect.getmembers(mod):
        #     if inspect.isclass(obj) and issubclass(obj, Plugin):
        #         print(f"Found plugin: {obj}")
        #         # Instantiate the plugin
        #         plugin = obj(dirpath=path)
        #         plugins.append(plugin)

        # TODO probably gonna have to detect by name instead (class & file name must be the same, and end with 'Plugin', e.g. StableDiffusionPlugin)
        for f in path.iterdir():
            if f.is_file() and f.suffix == '.py':
                mod = importlib.import_module(f'modules.{path.stem}.{f.stem}')
                for name, obj in inspect.getmembers(mod):
                    if inspect.isclass(obj) and issubclass(obj, Plugin):
                        # Instantiate the plugin
                        print(f"Found plugin: {obj}")
                        plugin = obj(dirpath=path)
                        plugins.append(plugin)
    except:
        sys.path.remove(path.as_posix())
        plugin_dirs.remove(path)


def load_all(loaddir: Path):
    """
    Load all plugin directories inside of loaddir.
    """
    if not loaddir.exists():
        return

    # Read the modules from the plugin directory
    for p in loaddir.iterdir():
        if p.is_dir() and not p.stem.startswith('__'):
            load(p)

    print(f'Reloaded modules, {len(plugin_dirs)} found')


def invoke(plugin, function, default=None, error=False, *args, **kwargs):
    """
    Invoke a plugin, may return a job object.
    """
    try:
        plug = get(plugin)
        if not plug:
            if error: printerr(f"Plugin {plugin} not found")
            return default

        attr = getattr(plug, function)
        if not attr:
            if error: printerr(f"Plugin {plugin} has no attribute {function}")
            return default

        return attr(*args, **kwargs)
    except Exception:
        print(f"Error calling: {plugin}.{function}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def job(p: JobParams):
    """
    Run a job.
    """
    plugin, funcname = p.get_plugin_impl()
    return invoke(plugin, funcname, None, True, p)


def broadcast(name, *args, **kwargs):
    """
    Dispatch a function call to all plugins.
    """
    print(f"broadcast({name}, {args}, {kwargs}) to {len(plugins)} plugins")
    for plugin in plugins:
        invoke(plugin.id, name, None, False, *args, **kwargs)
