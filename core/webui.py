import json
import threading

import core
from core import jobs
import modules
from flask import Flask, jsonify, request
import flask_socketio as fsock

queue_lock = threading.Lock()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = fsock.SocketIO(app)


def emit(event, *args, **kwargs):
    with queue_lock:
        socketio.emit(event, *args, **kwargs)


@app.route('/')
def index():
    return "Hello from stable-core!"


# A route to list all plugins
@app.route('/plugins')
def list_plugins():
    import core.plugins
    return jsonify(core.plugins.list_ids())


@socketio.on('connect')
def connect():
    print('Client connected')


@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')


@socketio.on('list_plugins')
def list_plugins():
    """ Send an array of plugin IDs """
    import core.plugins
    data = [extract_dict(x, 'id') for x in core.plugins.plugins]
    emit('list_plugins', jsonify(data))


@socketio.on('call_plugin')
def call_plugin(js):
    """
    An API message with socketio to call a plugin and optionally add a job
    """
    import core.plugins

    msg = json.loads(js)
    pid = msg['plugin_id']
    fname = msg['plugin_func']
    args = msg['args']
    kwargs = msg['kwargs']

    core.plugins.invoke(pid, fname, *args, **kwargs)


@socketio.on('list_jobs')
def list_jobs():
    return jsonify(jobs.queue.list_ids())


@socketio.on('is_job_processing')
def is_job_processing(id):
    return jsonify(jobs.is_processing(id))

@socketio.on('abort_job')
def abort_job(id):
    if jobs.is_processing(id):
        jobs.queue.abort(id)

@socketio.on('is_processing')
def is_processing():
    return jobs.is_any_processing()

def extract_dict(obj, *names):
    return {x: getattr(obj, x) for x in names}

# def wrap_gradio_gpu_call(func, extra_outputs=None):
#     def f(*args, **kwargs):
#         devicelib.torch_gc()
#
#         shared.state.sampling_step = 0
#         shared.state.job_count = -1
#         shared.state.job_no = 0
#         shared.state.job_timestamp = shared.state.get_job_timestamp()
#         shared.state.current_latent = None
#         shared.state.current_image = None
#         shared.state.current_image_sampling_step = 0
#         shared.state.skipped = False
#         shared.state.interrupted = False
#         shared.state.textinfo = None
#
#         with queue_lock:
#             res = func(*args, **kwargs)
#
#         shared.state.job = ""
#         shared.state.job_count = 0
#
#         devicelib.torch_gc()
#
#         return res
#
#     return gradio.ui.wrap_gradio_call(f, extra_outputs=extra_outputs)


# def launch_gradio():
#     while 1:
#         demo = gradio.ui.create_ui(wrap_gradio_gpu_call=wrap_gradio_gpu_call)
#
#         app, local_url, share_url = demo.launch(
#                 share=cmd_opts.share,
#                 server_name="0.0.0.0" if cmd_opts.listen else None,
#                 server_port=cmd_opts.port,
#                 debug=cmd_opts.gradio_debug,
#                 auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
#                 inbrowser=cmd_opts.autolaunch,
#                 prevent_thread_lock=True)
#
#         app.add_middleware(GZipMiddleware, minimum_size=1000)
#
#         while 1:
#             time.sleep(0.5)
#             if getattr(demo, 'do_restart', False):
#                 time.sleep(0.5)
#                 demo.close()
#                 time.sleep(0.5)
#                 break
#
#         print('Reloading modules: modules.ui')
#         importlib.reload(gradio.ui)
#
#         print('Restarting Gradio')
