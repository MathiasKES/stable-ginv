"""Best-effort filesystem helpers: chmod/makedirs/write/savefig and stdout tee."""
import os
import sys
from datetime import datetime


def safe_chmod(path, mode=0o770):
    """Set chmod on path, swallowing and logging errors."""
    try:
        os.chmod(path, mode)
    except OSError as e:
        print(f"[WARNING] Failed to chmod {path}: {e}")


def safe_makedirs(path, mode=0o770):
    """os.makedirs(exist_ok=True) with error swallowing. Returns True on success."""
    if not path:
        return True
    try:
        os.makedirs(path, mode=mode, exist_ok=True)
        return True
    except OSError as e:
        print(f"[WARNING] Failed to create directory {path}: {e}")
        return False


def safe_write(path, writer_fn, mode="w", newline=None, chmod_mode=0o770):
    """Open path for writing, call writer_fn(file). Best-effort chmod 770 on parent dir and file.
    Logs and returns False on any OSError instead of raising. Returns True on success.
    """
    parent = os.path.dirname(path)
    if parent and not safe_makedirs(parent, mode=chmod_mode):
        return False
    open_kwargs = {} if newline is None else {"newline": newline}
    try:
        with open(path, mode, **open_kwargs) as f:
            writer_fn(f)
    except OSError as e:
        print(f"[WARNING] Failed to write {path}: {e}")
        return False
    safe_chmod(path, mode=chmod_mode)
    return True


def safe_savefig(fig, path, chmod_mode=0o770, **savefig_kwargs):
    """Save matplotlib figure to path with best-effort chmod 770. Returns True on success."""
    parent = os.path.dirname(path)
    if parent and not safe_makedirs(parent, mode=chmod_mode):
        return False
    try:
        fig.savefig(path, **savefig_kwargs)
    except Exception as e:
        print(f"[WARNING] Failed to save figure {path}: {e}")
        return False
    safe_chmod(path, mode=chmod_mode)
    return True


def setstdout(ts=None, path=None):
    """Set up stdout tee to a log file. Returns the path used, or None if not interactive.

    ts:   timestamp string to name a new log file (ignored if path is given).
    path: full path to an existing log file to append to (worker processes pass this).
    If neither is given, a new file is created using datetime.now().
    """
    if os.environ.get("LSB_INTERACTIVE", default="N") != "Y":
        return None

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    terminal = sys.stdout

    if path is None:
        if ts is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists("/work3/s234843/bachelor/gpuout/idlg"):
            path = f"/work3/s234843/bachelor/gpuout/idlg/i{ts}.out"
        else:
            safe_makedirs("./gpuout")
            path = f"./gpuout/i{ts}.out"

    try:
        logfile = open(path, "a")
    except OSError as e:
        print(f"[WARNING] Failed to open stdout log {path}: {e}")
        return None
    safe_chmod(path)
    sys.stdout = Tee(terminal, logfile)
    return path
