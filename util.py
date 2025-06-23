import logging
import os
import subprocess
from collections import deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from tqdm import tqdm

PROGRESS = None
LOGFILE = Path("experiment.log").resolve()

logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@contextmanager
def timer(job, steps, detailed=False):
    logging.info(job)
    global PROGRESS
    if detailed:
        fmt = "{desc}: {n_fmt}/{total_fmt} {percentage:3.0f}%"
    else:
        fmt = "{desc}:{percentage:3.0f}%"

    PROGRESS = tqdm(
        total=steps,
        desc=job,
        position=0,
        bar_format=fmt,
        leave=False,
    )
    yield
    PROGRESS.close()


def inc_time_remaining(steps):
    PROGRESS.total += steps
    PROGRESS.refresh()


def update_timer(steps):
    if PROGRESS:
        PROGRESS.update(steps)


class CommandError(Exception):
    def __init__(self, desc, output):
        super().__init__(f"Command failed: {desc}\nOutput:\n{output}")


def command_runner(
    description="",
    write_progress=True,
    unit="objects",
    steps=None,
    verbose=False,
    allow_failure=True,
    dry_run=False,
):

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            logger = logging.getLogger()
            cmd = method(self, *args, **kwargs)
            desc = f"{description} {self.name}"
            with _temp_log_format(logger):
                if dry_run:
                    logger.info(f"\n[DRY RUN]\n")
                else:
                    logger.info(f"\n[RUNNING COMMAND]\n")
                if hasattr(self, "env"):
                    for key, value in self.env.items():
                        logging.info(f"{key}={value}")
                logger.info(f"{' '.join([str(x) for x in cmd])}\n")

            env = os.environ.copy()
            env.update(getattr(self, "env", {}))

            progress = steps or getattr(self, "steps", None)

            bar_format = (
                "{desc}: {n_fmt}/{total_fmt}"
                if getattr(self, "steps", False)
                else "{desc}:{percentage:3.0f}%"
            )

            if dry_run:
                return wrapper

            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            ) as proc:
                buffer = deque(maxlen=5)
                with tqdm(
                    total=progress,
                    desc=f"⤷ {desc}",
                    unit=unit,
                    leave=False,
                    position=1,
                    bar_format=bar_format,
                ) as pbar:
                    for line in proc.stdout:
                        line = line.rstrip()
                        logger.info(line)
                        buffer.append(line)
                        if verbose:
                            tqdm.write(line)
                        pbar.update(1)
                        update_timer(1)
            pbar.close()
            if not allow_failure and proc.returncode != 0:
                raise CommandError(desc, "\n".join(buffer))

        return wrapper

    return decorator


@contextmanager
def _temp_log_format(logger):
    handler = logger.handlers[0]
    if handler:
        original_format = handler.formatter
        handler.setFormatter(logging.Formatter("%(message)s"))
        yield
        handler.setFormatter(original_format)
    else:
        yield


def _terminate_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
