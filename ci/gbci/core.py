"""Constants, configuration and small utilities shared by every gbci module:
logging, JSON state files, subprocesses in their own process group, git,
flock, the CI phase record, CSV / text helpers."""
import contextlib
import csv
import datetime as dt
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent      # ci/gbci/core.py -> the checkout
STATE = Path(os.environ.get("GB_CI_STATE", ROOT / "ci-state"))
GPU_LOCK = ROOT / ".goldbach-gpu.lock"
RUN_RANGE = ROOT / "tools" / "run_range.sh"
BENCH = ROOT / "tools" / "bench.py"
WATERFALL = ROOT / "tools" / "waterfall.py"
UNIT = "goldbach-range"
ON_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

DEFAULTS = {
    # daemon branch watching ([] when GitHub Actions is the trigger)
    "watch_include": ["refs/heads/*"],
    "watch_exclude": ["refs/heads/exp/*", "refs/heads/wip/*"],
    "fetch": False,                 # also fetch + watch refs/remotes/origin/*
    "poll_s": 60,
    "build_jobs": max(2, (os.cpu_count() or 4) - 2),
    "werror": True,                 # -DGB_WERROR=ON, as the production build
    "gpu_validate": "quick",        # "quick", "full" or "" (skip)
    "bench_windows": "long,long-end",
    "bench_runs": 5,
    "bench_fail_pct": 2.0,          # candidate GPU time slower than active by this -> FAIL
    "bench_warn_pct": 0.5,
    # CI window: ci_slice integers from ci_window_start (default: the campaign
    # START, i.e. the first ci_slice integers of production slice 1 — same
    # segment geometry as production). "ci_slice": 0 = the last banked slice
    # in full, compared with its journal row (~28 min).
    "ci_slice": 20000000000000,
    "ci_window_start": 0,
    "slice_fail_pct": 10.0,         # CI window GPU time vs the reference result
    "slice_warn_pct": 1.5,
    # pausing production: stop at once — the slice in progress re-runs from
    # its start — when that redo is < this AND < the wait for its boundary
    "stop_max_loss_s": 300,
    "stop_wait_factor": 1.5,        # boundary wait <= this x median slice, then kill
    "slice_timeout_factor": 3.0,    # GB_RANGE_TIMEOUT for production slices
    "primecount": "",               # path; "" = PATH or ~/.local/src/primecount/build
    "primecount_threads": 4,
    "count_check": True,            # daemon: banked slices vs primecount
    "count_check_stop": True,       # ... and stop production on a mismatch
    "count_block": 200000000000000,  # integers per primecount call (consecutive slices)
    "auto_recheck": False,          # daemon: re-run queued retried slices
    "promote_diverged": False,      # deploy candidates that do not contain the active commit
    "max_attempts": 2,              # interrupted runs (reboot, crash) are retried up to this
    "keep_releases": 3,             # besides active / previous / journal-referenced ones
    # benchmark session after a PASS that changes code: "quick", "full" or ""
    "bench_on_promote": "quick",
    "bench_history": {"v2.1.0-dense": "ae73be3", "v1.13.0": "73ed9d3"},
}
BENCH_PROFILES = {
    # ~10 min: current (+ production config) vs previous release
    "quick": {"arms": ("cur", "prod", "prev"), "windows": "long,long-end", "runs": 5,
              "big": "w2e12=4000000000000000000:4000002000000000000", "big_runs": 3,
              "history": False, "sparse": False},
    # ~40 min: + GB_STATS=0, historical versions, sparse, every window
    "full": {"arms": ("cur", "prod", "nostats", "prev"), "windows": "bench,steady,end,long,long-end",
             "runs": 9, "big": "w2e12=4000000000000000000:4000002000000000000,"
                               "w2e12end=4999998000000000000:5000000000000000000",
             "big_runs": 3, "history": True, "sparse": True},
}
# csv keys that must equal the reference exactly (math or deterministic
# geometry); timings, ridges and the version may differ
EXACT_CSV = ["result", "leftovers", "segments", "primes", "twins", "stats_tail",
             "stats_sampled", "stats_pmax", "start", "limit"]
TIMING_KEYS = ["total_s", "gpu_s", "init_s", "sieve_s", "large_s", "check_s",
               "table_s", "sync_s", "evens_per_sec"]
STATS_STATEFUL = {"version", "new_first"}     # GBSTATS keys that depend on the accumulator
# GBSTATS fields a candidate may add over its reference release (compared once
# the reference writes them too); any other extra field fails
# the prime-gap fields (gstats.hpp gap_fields, written since v2.4): only
# relevant while the reference release predates them
STATS_NEW_OK = {"gaps", "gd1", "gmax", "gtail", "gwalk", "gsmom", "gsamp"}
NOTIFY_ENV = ("GB_NTFY_TOPIC", "GB_NOTIFY_MAIL", "GB_NOTIFY_CMD", "GB_NOTIFY_WALL",
              "GB_NOTIFY_SP", "GB_RECORD_HOOK")
FINAL = ("PASS", "FAIL", "IDENTICAL", "DIVERGED", "SKIPPED")
RESULT_COLS = (["run_id", "ts_start", "ts_end", "status", "reason", "refs", "commit", "version", "release",
                "previous_active", "segment", "window_start", "window_limit", "reference"]
               + ["slice_" + k for k in TIMING_KEYS]
               + ["slice_wall_s", "bench_long_speed_ratio", "bench_long-end_speed_ratio",
                  "checks_failed", "checks_warned", "downtime_s", "lost_s"])


class CIFail(Exception):
    """The candidate failed a gate (FAIL); any other exception is an ERROR."""


# ------------------------------------------------------------------ utils --
def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def stamp():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def log(msg):
    line = f"{now_iso()} {msg}"
    print(line, flush=True)
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "ci.log", "a") as f:
        f.write(line + "\n")


_last_tb = {}


def log_exc(line, key=None):
    """log(line), then the traceback of the exception being handled. With a
    key, a traceback identical to that key's previous one is not repeated
    (the daemon hitting the same error on every poll); line still is."""
    tb = traceback.format_exc().rstrip()
    if key is not None:
        if _last_tb.get(key) == tb:
            log(line + " (same traceback as before)")
            return
        _last_tb[key] = tb
    log(line + "\n" + tb)


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def save_json(p, obj, mode=0o644):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, p)


def cfg():
    c = dict(DEFAULTS)
    c.update(load_json(STATE / "config.json", {}))
    return c


def killpg(p):
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def sh(cmd, cwd=None, env=None, logf=None, timeout=None, check=False):
    """Run cmd in its own process group (killed as a whole on timeout or when
    the CI itself is interrupted); output appended to logf. -> (rc, output)."""
    if logf:
        with open(logf, "a") as f:
            f.write(f"\n$ {' '.join(shlex.quote(str(c)) for c in cmd)}\n")
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        killpg(p)
        out, _ = p.communicate()
        rc, out = 124, (out or "") + f"\nTIMEOUT after {timeout}s\n"
    except BaseException:
        killpg(p)
        raise
    if logf:
        with open(logf, "a") as f:
            f.write(out)
    if check and rc != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed rc={rc}: {out.strip()[-400:]}")
    return rc, out


def git(*args, check=True):
    p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()


def is_ancestor(a, b):
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", a, b]).returncode == 0


def kv(line):
    return dict(t.split("=", 1) for t in (line or "").split() if "=" in t)


def plot_python():
    py = ROOT / "tools/.venv/bin/python"
    return str(py) if py.exists() else sys.executable


@contextlib.contextmanager
def flock(path, blocking=False, timeout=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    t0 = time.monotonic()
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not blocking or time.monotonic() - t0 > timeout:
                f.close()
                raise RuntimeError(f"{path} is held by another process")
            time.sleep(1)
    try:
        yield f
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def set_phase(state, phase, **extra):
    """Record the CI phase (crash recovery, dashboard, heartbeat)."""
    state.update(phase=phase, phase_since=time.time(), **extra)
    save_json(STATE / "state.json", state)


def read_lines(path):
    """The lines of a text file (OSError when it cannot be read)."""
    with open(path) as f:
        return f.read().splitlines()


def csv_rows(path):
    """Every row of a CSV file as a dict; [] when the file does not exist."""
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def csv_header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f), None)


def hms(s):
    s = int(max(s or 0, 0))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"
