import uuid
from abc import abstractmethod
from datetime import datetime

from tqdm import tqdm

# IDEAS
# - We could have other stable-core nodes and dispatch to them (probably better ways to do this lol)
from core import webui
from core.cmdargs import cargs
from core.printing import progress_print_out


class JobParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @abstractmethod
    def get_plugin_impl(self):
        pass

    def plugin(self):
        return self.job.plugin()

    def on_start(self, job):
        pass


class Job:
    def __init__(self, plugin_id: str, plugin_func: str, parameters: JobParams):
        self.job_id = str(uuid.uuid4())
        self.plugin_id = plugin_id
        self.plugin_func = plugin_func
        # State
        self.p: JobParams = parameters
        if self.p is not None:
            self.p.job = self

        self.state: str = ""
        self.progress: float = 0
        self.aborted: bool = False
        self.skipped: bool = False
        self.step: int = 0
        self.stepmax: int = 0
        self.latent = None
        self.image = None
        self.timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S")  # shouldn't this return job_timestamp?

    def plugin(self):
        import core.plugins
        return core.plugins.get(self.plugin_id)

    def done(self):
        return self.progress == 1

    def update(self, progress):
        self.progress = progress
        webui.emit('updated_job', self.job_id, self.progress)

    def update_step(self, num=None):
        if num is None:
            num = self.step + 1

        self.step = num
        self.progress = self.step / self.stepmax

        tqdm_total.update()
        # if opts.show_progress_every_n_steps > 0:

    def __repr__(self):
        return f"Job({self.job_id}, {self.plugin_id}, {self.plugin_func}, {self.progress})"


class JobQueue:
    def __init__(self):
        self.all = []  # job objects
        self.queued = []  # job objects
        self.processing = []  # job objects

    def list(self):
        return dict(all=self.all, queud=self.queued, processing=self.processing)

    def enqueue(self, job):
        self.all.append(job)
        self.queued.append(job)

        webui.emit('added_job', job.job_id)

    def process(self, job):
        """
        Must already be enqueued
        """
        if job not in self.queued:
            raise Exception("Job not in queue")

        self.queued.remove(job)
        self.processing.append(job)

        job.p.on_start(job)

        webui.emit('started_job', job.job_id)

    def cancel(self, job):
        if job not in self.queued:
            raise Exception("Job not in queue")

        self.queued.remove(job)
        self.all.remove(job)

        webui.emit('cancelled_job', job.job_id)

    def finish(self, job):
        if job not in self.processing:
            raise Exception("Job not in processing")

        self.processing.remove(job)
        self.all.remove(job)

        webui.emit('finished_job', job.job_id)

    def abort(self, job):
        if job not in self.processing:
            raise Exception("Job not in processing")

        self.processing.remove(job)
        self.all.remove(job)

        job.aborted = True

        webui.emit('aborted_job', job.job_id)

    def remove(self, job):
        if job not in self.all:
            raise Exception("Job not in queue")

        if job in self.queued:
            self.cancel(job)

        if job in self.processing:
            self.abort(job)

        self.all.remove(job)

        webui.emit('removed_job', job.job_id)


class JobTQDM:
    def __init__(self):
        self.tqdm = None

    def create(self):
        self.tqdm = tqdm(desc="Total progress",
                         total=queue.all,
                         position=1,
                         file=progress_print_out)

    def update(self):
        from core.options import opts
        if not opts.multiple_tqdm or cargs.disable_console_progressbars:
            return

        if self.tqdm is None:
            self.create()

        self.tqdm.update()

    def update_total(self, new_total):
        from core.options import opts
        if not opts.multiple_tqdm or cargs.disable_console_progressbars:
            return
        if self.tqdm is None:
            self.create()
        self.tqdm.total = new_total

    def clear(self):
        if self.tqdm is not None:
            self.tqdm.close()
            self.tqdm = None


def get(job_id):
    """
    Get a job by id.
    The job must be queued or processing.
    """
    if isinstance(job_id, Job):
        return job_id

    for job in queue.all:
        if job.job_id == job_id:
            return job
    raise Exception("Job not found")


def finish(job_id):
    job = get(job_id)

    job.update(1)
    queue.finish(job)
    start_next()


def start_next():
    """
    Start the next job in the queue
    """
    if len(queue.queued) == 0:
        return

    job = queue.queued[0]
    queue.process(job)


def enqueue(job):
    """
    Add a job to the queue.
    """
    queue.enqueue(job)


def new_job(plugid, name, jobparams):
    global total_created
    total_created += 1
    j = Job(plugid, name, jobparams)
    enqueue(j)

    return j


# We need to do the job's queue in global functions to keep the api clean
queue = JobQueue()
total_created = 0
tqdm_total = JobTQDM()


def is_processing(id):
    """
    Check if a job is processing (by ID)
    """
    j = get(id)
    if j is not None:
        return j in queue.processing


def is_any_processing():
    return len(queue.processing) > 0