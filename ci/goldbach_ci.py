#!/usr/bin/env python3
"""goldbach_ci.py — local CI for goldbach-l0 (stdlib only).

One CI run per commit (GitHub Actions job `gpu`, or the daemon), see
ci/README.md:

  build in a separate worktree (+ CPU tests unless Actions already ran them)
    -> same code as the active release?  IDENTICAL: done (active unchanged)
    -> pause run_range (slice boundary, or now if the slice just started)
    -> tests/gpu_validate.sh quick -> bench.py A/B vs the active release
    -> CI window (fixed 2e13 integers at the campaign start, production args
       and env, isolated statistics files) with the candidate; compared bit
       for bit with the active release's result for the same window (cached
       per release) and with primecount
    -> PASS: promote (previous kept for rollback), quick benchmark session
       FAIL / ERROR: the active release stays
    -> resume run_range with the active release; resume point verified

State: ci-state/ (gitignored) — releases/<id>/, active + previous symlinks,
ledger.jsonl + results.csv, runs/<id>/ (logs, report.md, figures),
production.json (the run_range invocation, may hold secrets: 0600),
state.json (current phase: crash recovery, dashboard), queue.json (runs to
retry), ref-cache/ (reference window results per release). GPU work is
serialised by an flock on .goldbach-gpu.lock, held by tools/run_range.sh for
its whole life; CI runs are serialised by ci-state/ci.lock.
"""
import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
STATS_NEW_OK = {"gaps", "gd1", "gmax", "gtail", "gwalk", "gsmom", "gsamp"}     # v2.4 prime gaps (gstats.hpp gap_fields)
NOTIFY_ENV = ("GB_NTFY_TOPIC", "GB_NOTIFY_MAIL", "GB_NOTIFY_CMD", "GB_NOTIFY_WALL",
              "GB_NOTIFY_SP", "GB_RECORD_HOOK")
# run_range.sh's defaults for the result-relevant GB_* settings (goldbach-tools
# keeps the authoritative copy): every CI slice, reference key and "prod"
# benchmark arm runs as production does. GB_GAPS=sampled: +4.9% (gap histogram
# and moments from the sampled segments only); GB_BUCKET_VERIFY=1000: +1.1%
# (PR #13's B580 runs, 2026-11-14).
PROD_ENV = {"GB_BUCKET_VERIFY": "1000", "GB_GAPS": "sampled"}
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


def alert(title, msg):
    """Notification hook (scripts/notify_record.sh ALERT, production's env)."""
    hook = os.environ.get("GB_RECORD_HOOK", "bash scripts/notify_record.sh")
    if not hook:
        return
    env = dict(os.environ)
    env.update(prod_conf().get("env", {}))
    subprocess.Popen(shlex.split(hook) + ["ALERT", title, msg, "-", "-"], cwd=ROOT, env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


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


# ----------------------------------------------------------- journal ------
def read_journal(path):
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return rows
    for ln in lines:
        t = ln.split()
        if len(t) >= 4 and t[0] in ("PASS", "RETRY"):
            if not (t[1].isdigit() and t[2].isdigit()):   # e.g. a row cut off by a crash
                print(f"warning: malformed journal row ignored: {ln[:120]}", file=sys.stderr)
                continue
            rows.append({"tag": t[0], "a": int(t[1]), "b": int(t[2]), "ts": t[3],
                         "kv": kv(" ".join(t[4:])), "line": ln})
    return rows


def chain_next(rows, start):
    """run_range.sh's resume rule: follow the PASS chain from START.

    A row extends the chain only if its own verifier output agrees with its
    slice fields (start= / limit= of the --csv line): a damaged b would
    otherwise skip unverified evens. Disagreement raises (fail closed)."""
    nxt = start
    for r in rows:
        if r["tag"] == "PASS" and r["a"] == nxt:
            if r["kv"].get("start") != str(r["a"]) or r["kv"].get("limit") != str(r["b"]):
                raise RuntimeError(f"journal PASS row for [{r['a']}, {r['b']}] disagrees with its own "
                                   f"--csv start=/limit= (damaged journal?): {r['line'][:160]}")
            nxt = r["b"] + 2
    return nxt


def journal_audit(rows, start):
    """(resume point, duplicate PASS rows) of the journal."""
    seen, dups = set(), []
    for r in rows:
        if r["tag"] == "PASS":
            k = (r["a"], r["b"])
            if k in seen:
                dups.append(k)
            seen.add(k)
    return chain_next(rows, start), dups


def median_slice_wall(rows, n=20):
    ts = [dt.datetime.fromisoformat(r["ts"]) for r in rows if r["tag"] == "PASS"]
    d = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    d = [x for x in d[-n:] if 60 < x < 6 * 3600]
    return st.median(d) if d else 1800.0


def stats_line(path, a, b):
    """Last GBSTATS line of a goldbach_stats.log for exactly [a, b]."""
    hit = None
    with contextlib.suppress(OSError):
        with open(path) as f:
            for ln in f:
                if ln.startswith("GBSTATS") and f" start={a} " in ln and f" limit={b} " in ln:
                    hit = ln.strip()
    return hit


# -------------------------------------------------------------- releases --
def fingerprint(rel):
    """Code fingerprint: the binary without debug info / build-id (both embed
    the build path) + every shipped kernel module. Equal fingerprints = the
    same machine code and kernels, whatever directory they were built in."""
    h = hashlib.sha256()
    with tempfile.TemporaryDirectory() as td:
        s = Path(td) / "s"
        subprocess.run(["strip", "--remove-section=.note.gnu.build-id", "-o", str(s),
                        str(rel / "goldbach_l0")], check=True)
        h.update(s.read_bytes())
    for k in sorted((rel / "kernels").glob("*.spv")):
        h.update(k.name.encode())
        h.update(k.read_bytes())
    return h.hexdigest()


def release_dir(rid):
    return STATE / "releases" / rid


def link_target(name):
    p = STATE / name
    return p.resolve() if p.exists() else None


def active_release():
    return link_target("active")


def release_meta(rel):
    return load_json(rel / "meta.json", {}) if rel else {}


def set_link(name, target):
    tmp = STATE / f".{name}.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(os.path.relpath(target, STATE), tmp)
    os.replace(tmp, STATE / name)


def promote(rel):
    cur = active_release()
    # previous = the last DIFFERENT code: a code-identical promotion (docs,
    # CI scripts) must not push the real rollback target out
    if cur and cur != rel.resolve() and \
            release_meta(cur).get("fingerprint") != release_meta(rel).get("fingerprint"):
        set_link("previous", cur)
    set_link("active", rel)
    prev = link_target("previous")
    log(f"promoted: active = {rel.name} (previous = {prev.name if prev else '-'})")
    prune_releases()


def prune_releases():
    """Keep active, previous, every release named in the journal, the newest
    keep_releases others and the binary production runs right now."""
    keep = {p for p in (active_release(), link_target("previous")) if p}
    pc = prod_conf()
    if pc:
        keep |= {release_dir(r["kv"]["release"]).resolve() for r in read_journal(prod_paths(pc)[0])
                 if "release" in r["kv"]}
    for p in find_run_range():
        env = Path(f"/proc/{p['pid']}/environ")
        with contextlib.suppress(OSError):
            for e in env.read_bytes().split(b"\0"):
                if e.startswith(b"GB_RANGE_BIN="):
                    keep.add(Path(e.split(b"=", 1)[1].decode()).parent.resolve())
    rels = sorted((d for d in (STATE / "releases").glob("*") if d.is_dir() and not d.name.endswith(".staging")),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    spare = [d for d in rels if d.resolve() not in keep][cfg()["keep_releases"]:]
    for d in spare:
        shutil.rmtree(d, ignore_errors=True)
    if spare:
        log(f"pruned releases: {', '.join(d.name for d in spare)}")


# ------------------------------------------------------------- production --
def proc_cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
    except OSError:
        return []


def find_run_range():
    """Top-level run_range.sh processes: [{pid, pgid, args, unit}]."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        cl = [c.decode(errors="replace") for c in proc_cmdline(d.name)]
        if len(cl) < 2 or Path(cl[0]).name not in ("bash", "sh") or not cl[1].endswith("run_range.sh"):
            continue
        try:
            ppid = int(Path(f"/proc/{d.name}/stat").read_text().rsplit(")", 1)[1].split()[1])
            pgid = os.getpgid(int(d.name))
        except (OSError, IndexError, ValueError):
            continue
        pcl = proc_cmdline(ppid)
        if len(pcl) >= 2 and pcl[1].decode(errors="replace").endswith("run_range.sh"):
            continue                       # a $(...) subshell of another instance
        try:
            cg = Path(f"/proc/{d.name}/cgroup").read_text()
        except OSError:
            cg = ""
        out.append({"pid": int(d.name), "pgid": pgid, "args": cl[2:],
                    "unit": UNIT if f"/{UNIT}.service" in cg else ""})
    return out


def pgrep(name):
    p = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
    return [int(x) for x in p.stdout.split()]


def prod_conf():
    return load_json(STATE / "production.json", {})


def save_prod_conf(pc):
    save_json(STATE / "production.json", pc, mode=0o600)     # env may hold secrets


def prod_paths(pc):
    j = ROOT / pc["journal"]
    base = str(j)[: -len(".journal")] if str(j).endswith(".journal") else str(j)
    return j, Path(base + ".log"), Path(base + ".stop")


def prod_stats_log(pc):
    """Production's goldbach_stats.log (GB_STATS_LOG, relative to ROOT where
    run_range runs)."""
    return ROOT / pc.get("env", {}).get("GB_STATS_LOG", "goldbach_stats.log")


def prod_live(pc):
    """Production's goldbach_live.json if it is fresh (< 2 min old)."""
    p = Path(pc.get("env", {}).get("GB_STATS_LIVE", ROOT / "goldbach_live.json"))
    lv = load_json(p, {})
    return lv if lv and time.time() - lv.get("time", 0) < 120 else None


def adopt_production():
    """Record the running run_range invocation (args, GB_* env, nice)."""
    procs = find_run_range()
    if not procs:
        raise RuntimeError("no run_range.sh process to adopt")
    p = procs[0]
    args = p["args"]
    start, limit = args[0], args[1]
    slice_ = args[2] if len(args) > 2 else "200000000000000"
    journal = args[3] if len(args) > 3 else f"range-{start}-{limit}.journal"
    env = {}
    with contextlib.suppress(OSError):
        for e in Path(f"/proc/{p['pid']}/environ").read_bytes().split(b"\0"):
            k, _, v = e.decode(errors="replace").partition("=")
            if k.startswith("GB_") and not k.startswith("GB_RANGE_BIN") and k != "GB_RANGE_SPV":
                env[k] = v
    nice = int(Path(f"/proc/{p['pid']}/stat").read_text().rsplit(")", 1)[1].split()[16])
    pc = prod_conf()
    pc.update({"start": start, "limit": limit, "slice": slice_, "journal": journal,
               "env": env, "nice": nice, "desired": "running"})
    save_prod_conf(pc)
    log(f"adopted production: run_range {start} {limit} {slice_} {journal} nice={nice} env={sorted(env)}")
    return pc


def production_status():
    pc = prod_conf()
    procs = find_run_range()
    s = {"running": bool(procs), "procs": procs, "desired": pc.get("desired", "?")}
    if pc:
        rows = read_journal(prod_paths(pc)[0])
        s["next"] = chain_next(rows, int(pc["start"]))
        s["banked"] = sum(r["tag"] == "PASS" for r in rows)
    return s


def stop_now_is_cheaper(age, frac, max_loss):
    """Stop at once (redo `age` s of GPU work) rather than wait for the slice
    boundary, iff that redo is below max_loss and below the remaining wait."""
    remaining = age * (1 - frac) / frac if frac > 0 else float("inf")
    return age < max_loss and age < remaining


def stop_production(now=False, reason=""):
    """Stop run_range without losing completed work: at the next slice
    boundary (stop file), or at once if `now` or the slice in progress is
    younger than stop_max_loss_s (it simply re-runs). -> info dict."""
    c = cfg()
    procs = find_run_range()
    info = {"was_running": bool(procs), "mode": "now" if now else "boundary", "lost_s": 0}
    if not procs:
        stray = pgrep("goldbach_l0")
        if stray:
            raise RuntimeError(f"goldbach_l0 running without run_range (pid {stray}); not touching it")
        return info
    pc = prod_conf() or adopt_production()
    j, lg, stopf = prod_paths(pc)
    rows = read_journal(j)
    info["next_before"] = chain_next(rows, int(pc["start"]))
    lv = prod_live(pc)
    age = lv.get("elapsed_s") if lv and lv.get("start") == info["next_before"] else None
    if not now and age is not None and stop_now_is_cheaper(age, lv["evens_done"] / max(lv["evens_total"], 1),
                                                           c["stop_max_loss_s"]):
        now, info["mode"] = True, f"now (slice in progress only {age:.0f}s old; it re-runs)"
    if now:
        info["lost_s"] = int(age or 0)
    log(f"stopping production ({info['mode']}{': ' + reason if reason else ''}); "
        f"resume point {info['next_before']}")

    def kill_all(sig):
        for p in find_run_range():
            if p["unit"]:
                subprocess.run(["systemctl", "--user", "kill", "--signal=" + sig.name, UNIT + ".service"],
                               capture_output=True)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p["pgid"], sig)
        for pid in pgrep("goldbach_l0"):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, sig)

    def wait_gone(sec):
        t0 = time.monotonic()
        while time.monotonic() - t0 < sec:
            if not find_run_range() and not pgrep("goldbach_l0"):
                return True
            time.sleep(1)
        return False

    if now:
        kill_all(signal.SIGTERM)
    else:
        limit_s = c["stop_wait_factor"] * median_slice_wall(rows) + 120
        t0 = time.monotonic()
        while find_run_range():
            stopf.touch()                    # re-assert: a fresh start removes stale files
            if time.monotonic() - t0 > limit_s:
                log(f"boundary wait timed out after {limit_s:.0f}s; killing the slice in progress")
                lv = prod_live(pc)
                info["lost_s"] = int(lv.get("elapsed_s", 0)) if lv else -1
                kill_all(signal.SIGTERM)
                break
            time.sleep(2)
    if not wait_gone(60):
        kill_all(signal.SIGKILL)
        if not wait_gone(30):
            raise RuntimeError("run_range / goldbach_l0 did not exit")
    with contextlib.suppress(FileNotFoundError):
        stopf.unlink()
    subprocess.run(["systemctl", "--user", "reset-failed", UNIT + ".service"], capture_output=True)
    rows = read_journal(j)
    info["next_after"], dups = journal_audit(rows, int(pc["start"]))
    info["banked_after"] = sum(r["tag"] == "PASS" for r in rows)
    if dups:
        raise RuntimeError(f"journal has duplicate PASS rows: {dups[:3]}")
    if info["next_after"] < info["next_before"]:
        raise RuntimeError("journal resume point went backwards")
    log(f"production stopped; resume point {info['next_after']} ({info['banked_after']} banked"
        + (f", {info['lost_s']}s of a partial slice to redo" if info["lost_s"] else "") + ")")
    return info


def last_run_range_event(lg):
    try:
        with open(lg, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    for ln in reversed(tail):
        if re.search(r" (STOPPED|COMPLETE|FATAL|STOP requested|run_range:)", ln) or ln.startswith("FATAL"):
            return ln
    return ""


def start_production(expect_release=None):
    """Start run_range under the systemd user unit with the active release and
    verify its 'resuming at N release=R' line against the journal chain."""
    pc = prod_conf()
    if not pc:
        raise RuntimeError("no production.json (run `production adopt` or bootstrap first)")
    if find_run_range() or pgrep("goldbach_l0"):
        raise RuntimeError("run_range / goldbach_l0 already running")
    rel = active_release()
    if not rel:
        raise RuntimeError("no active release")
    expect_release = expect_release or release_meta(rel).get("id")
    j, lg, stopf = prod_paths(pc)
    rows = read_journal(j)
    expect_next = chain_next(rows, int(pc["start"]))
    env = dict(pc.get("env", {}))
    env["GB_RANGE_BIN"] = str(rel / "goldbach_l0")
    env.setdefault("GB_RANGE_TIMEOUT", str(int(cfg()["slice_timeout_factor"] * median_slice_wall(rows))))
    with contextlib.suppress(FileNotFoundError):
        stopf.unlink()
    subprocess.run(["systemctl", "--user", "reset-failed", UNIT + ".service"], capture_output=True)
    base = ["systemd-run", "--user", "--collect", f"--unit={UNIT}",
            "--description=goldbach-l0 run_range (local CI managed)", f"--working-directory={ROOT}",
            # the site sync runs whenever production does (both stay disabled: no autostart at boot)
            "--property=Wants=goldbach-sync.timer goldbach-sync.path"]
    base += [f"--setenv={k}={v}" for k, v in sorted(env.items())]
    cmd = ["/usr/bin/bash", str(RUN_RANGE), pc["start"], pc["limit"], pc["slice"], pc["journal"]]
    size0 = lg.stat().st_size if lg.exists() else 0
    nice = pc.get("nice", 0)
    p = subprocess.run(base + ([f"--property=Nice={nice}"] if nice else []) + cmd, capture_output=True, text=True)
    if p.returncode != 0 and nice and "Nice" in p.stderr:
        log(f"systemd-run with Nice={nice} refused ({p.stderr.strip()}); starting at default priority")
        subprocess.run(["systemctl", "--user", "reset-failed", UNIT + ".service"], capture_output=True)
        p = subprocess.run(base + cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"systemd-run failed: {p.stderr.strip()}")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 60:
        time.sleep(1)
        try:
            with open(lg) as f:
                f.seek(size0)
                new = f.read()
        except OSError:
            new = ""
        if "FATAL" in new:
            raise RuntimeError("run_range refused to start: " + new.strip().splitlines()[-1])
        m = re.search(r"run_range: .* resuming at (\d+) release=(\S+)", new)
        if m:
            got, relid = int(m.group(1)), m.group(2)
            if got != expect_next:
                raise RuntimeError(f"run_range resumed at {got}, journal chain says {expect_next}")
            if relid != expect_release:
                raise RuntimeError(f"run_range runs release {relid}, expected {expect_release}")
            pc["desired"] = "running"
            save_prod_conf(pc)
            log(f"production resumed (unit {UNIT}): release {relid}, resume point {got} (verified)")
            return {"resumed_at": got, "release": relid}
    raise RuntimeError("run_range did not report its resume point within 60 s")


@contextlib.contextmanager
def production_paused(reason, state=None):
    """Production stopped (if it ran) for the body, resumed afterwards — also
    after a failure or SIGTERM. Yields the stop info."""
    if state is not None:
        set_phase(state, "stop_production")
    info = stop_production(reason=reason)
    if state is not None:
        state["prod_stopped"] = info["was_running"]
        save_json(STATE / "state.json", state)
    try:
        yield info
    finally:
        if info["was_running"] and not (state or {}).get("no_resume"):
            if state is not None:
                set_phase(state, "resume_production")
            try:
                info.update(start_production())
            except Exception as e:
                info["resume_error"] = str(e)
                log(f"could NOT resume production: {e}")
                alert("CI could not resume run_range", str(e))


# -------------------------------------------------------------- primecount --
def primecount_bin():
    for cand in [cfg()["primecount"], shutil.which("primecount") or "",
                 str(Path.home() / ".local/src/primecount/build/primecount")]:
        if cand and os.access(cand, os.X_OK):
            return cand
    return None


def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3e24 (first 12 prime bases)."""
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def pi(x):
    """pi(x) via primecount, cached in ci-state/pi-cache.json; pi(odd x)
    reuses a cached pi(x - 1)."""
    cache = load_json(STATE / "pi-cache.json", {})
    if str(x) in cache:
        return int(cache[str(x)])
    if x % 2 and str(x - 1) in cache:
        return int(cache[str(x - 1)]) + int(is_prime(x))
    pc = primecount_bin()
    if not pc:
        raise RuntimeError("primecount not found (set primecount in ci-state/config.json)")
    p = subprocess.run(["nice", "-n", "19", pc, str(x), f"-t{cfg()['primecount_threads']}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"primecount {x}: {p.stderr.strip()}")
    v = int(p.stdout.split()[-1])
    cache = load_json(STATE / "pi-cache.json", {})
    cache[str(x)] = str(v)
    save_json(STATE / "pi-cache.json", cache)
    return v


def primes_in(a, b):
    """Number of primes in [a, b]."""
    return pi(b) - pi(a - 1)


# ------------------------------------------------------- system / GPU stack --
GPU_STACK_PKGS = ("intel-compute-runtime", "intel-graphics-compiler", "level-zero-loader")


def gpu_stack():
    """What turns the shipped SPIR-V into GPU machine code and runs it: the
    kernel + xe module, the compute runtime / IGC / Level Zero packages, the
    VRAM clock. A change here changes production's GPU code (or its memory
    timing) without any commit."""
    snap = {"kernel": os.uname().release}
    with contextlib.suppress(OSError):
        snap["xe"] = Path("/sys/module/xe/srcversion").read_text().strip()
    p = subprocess.run(["pacman", "-Q", *GPU_STACK_PKGS], capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        name, ver = ln.split()[:2]
        snap[name] = ver
    for f in Path("/sys/class/drm").glob("card[0-9]*/device/tile0/gt0/oc/mem_speed"):
        with contextlib.suppress(OSError):
            snap["mem_speed"] = f.read_text().strip()
    return snap


GPU_STACK_FAILED = "gpu-stack-failed.json"


def gpu_stack_change(now=None):
    """{key: (verified, now)} since the active release was last verified."""
    ref = load_json(STATE / "gpu-stack.json", None)
    now = gpu_stack() if now is None else now
    if ref is None:                      # first run: today's stack is the baseline
        save_json(STATE / "gpu-stack.json", now)
        return {}
    return {k: (ref.get(k), now.get(k)) for k in sorted(set(ref) | set(now)) if ref.get(k) != now.get(k)}


def gpu_stack_check_due():
    """The stack change still to verify: {} when there is none, or when the
    re-verification already failed on exactly this stack (ci-state/
    gpu-stack-failed.json; it is retried once the stack changes again, or
    after that file is deleted). Without this, a failed check re-ran -- and
    alerted -- on every daemon poll."""
    now = gpu_stack()
    change = gpu_stack_change(now)
    if change and load_json(STATE / GPU_STACK_FAILED, None) == now:
        return {}
    return change


# ------------------------------------------------------------------ build --
def tools_worktree(logf):
    """goldbach-tools for CI builds: a clean detached worktree at the tools
    repo's committed HEAD (the worktree's tools -> ../goldbach-tools resolves
    here), never the live working tree, so a verdict depends on commits only.
    -> the tools commit used."""
    live = (ROOT / "tools").resolve()
    tl = STATE / "work" / "goldbach-tools"
    head = subprocess.run(["git", "-C", str(live), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(live), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if tl.is_symlink():                         # layout before v2.3.0: a link to the live tree
        tl.unlink()
    if not (tl / ".git").exists():
        shutil.rmtree(tl, ignore_errors=True)
        sh(["git", "-C", live, "worktree", "prune"], logf=logf)
        sh(["git", "-C", live, "worktree", "add", "--detach", tl, head], logf=logf, check=True)
    else:
        sh(["git", "-C", tl, "checkout", "--detach", "-f", head], logf=logf, check=True)
    sh(["git", "-C", tl, "clean", "-ffdxq"], logf=logf, check=True)
    log(f"tools: goldbach-tools {head[:12]}" +
        (" (the live tools tree has uncommitted changes: not used)" if dirty else ""))
    return head


def worktree(sha, logf):
    wt = STATE / "work" / "src"
    (STATE / "work").mkdir(parents=True, exist_ok=True)
    tools_worktree(logf)
    if not (wt / ".git").exists():
        shutil.rmtree(wt, ignore_errors=True)
        git("worktree", "prune")
        sh(["git", "-C", ROOT, "worktree", "add", "--detach", wt, sha], logf=logf, check=True)
    else:
        sh(["git", "-C", wt, "checkout", "--detach", "-f", sha], logf=logf, check=True)
    sh(["git", "-C", wt, "clean", "-ffdxq"], logf=logf, check=True)
    for f in list(ROOT.glob("base_primes*.cache*")) + list(ROOT.glob("scm_plan.*.bin")):
        if not (wt / f.name).exists():
            os.symlink(f, wt / f.name)
    return wt


def kernel_freshness(wt, logf):
    """Every shipped kernels/<label>_bmg.spv must equal a fresh ocloc compile
    (the GitHub `kernels` gate, here when ocloc is the pinned version)."""
    ci = (wt / ".github/workflows/ci.yml").read_text()
    pinned = re.search(r"CR_VER:\s*([\d.]+)", ci)
    mods = re.search(r"KERNEL_MODULES: \|\n((?: {4}\S.*\n)+)", ci)
    oc = shutil.which("ocloc")
    if not oc or not pinned or not mods:
        return "warn", "ocloc or the module list not available: kernel freshness not checked"
    ver = subprocess.run([oc, "--version"], capture_output=True, text=True).stdout
    if pinned.group(1).rsplit(".", 1)[0] not in ver:
        return "warn", f"local ocloc ({ver.strip()[:40]}) != pinned {pinned.group(1)}: not checked"
    bad = []
    with tempfile.TemporaryDirectory() as td:
        for ln in mods.group(1).splitlines():
            label, src, opts = (ln.strip().split("|") + ["", ""])[:3]
            cmd = ["nice", "-n", "15", oc, "compile", "-file", wt / src, "-device", "bmg", "-output", f"{td}/{label}"]
            rc, _ = sh(cmd + (["-options", opts] if opts else []), cwd=td, logf=logf)
            fresh, shipped = Path(td) / f"{label}_bmg.spv", wt / "kernels" / f"{label}_bmg.spv"
            if rc != 0 or not fresh.exists() or not shipped.exists() or fresh.read_bytes() != shipped.read_bytes():
                bad.append(label)
    return ("fail", f"stale or broken kernel modules: {bad}") if bad else ("ok", "all shipped .spv fresh")


def build_release(sha, run_dir, refs, cpu_tests=True):
    """Build sha in the CI worktree (+ kernel freshness and ctest -LE gpu
    unless the caller already ran them) and stage it as a release.
    -> (worktree, release dir or None, checks)."""
    c = cfg()
    logf = run_dir / "build.log"
    wt = worktree(sha, logf)
    checks = []
    cm = ["cmake", "-S", wt, "-B", wt / "build", "-DCMAKE_BUILD_TYPE=Release"] + (["-DGB_WERROR=ON"] if c["werror"] else [])
    rc, _ = sh(["nice", "-n", "15", *cm], logf=logf)
    if rc == 0:
        rc, _ = sh(["nice", "-n", "15", "cmake", "--build", wt / "build", "-j", str(c["build_jobs"])],
                   logf=logf, timeout=3600)
    checks.append({"name": "build", "level": "ok" if rc == 0 else "fail",
                   "detail": "cmake Release" + (" -Werror" if c["werror"] else "") + f" rc={rc}"})
    if rc != 0:
        return wt, None, checks
    if cpu_tests:
        lvl, det = kernel_freshness(wt, run_dir / "kernels.log")
        checks.append({"name": "kernels_fresh", "level": lvl, "detail": det})
        rc, out = sh(["nice", "-n", "15", "ctest", "--test-dir", wt / "build", "-LE", "gpu",
                      "--output-on-failure", "--timeout", "900"], logf=run_dir / "ctest.log", timeout=3600)
        summ = next((ln for ln in out.splitlines() if "tests passed" in ln), out.strip()[-200:])
        checks.append({"name": "ctest_cpu", "level": "ok" if rc == 0 else "fail", "detail": summ.strip()})
        if any(x["level"] == "fail" for x in checks):
            return wt, None, checks
    else:
        checks.append({"name": "cpu_tests", "level": "ok",
                       "detail": "kernel freshness + ctest ran in the GitHub jobs kernels / host"})
    rid = sha[:12]
    rel = release_dir(rid)
    stage = rel.with_name(rid + ".staging")
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "kernels").mkdir(parents=True)
    for f in ("goldbach_l0", "stats_ref"):
        if (wt / "build" / f).exists():
            shutil.copy2(wt / "build" / f, stage / f)
    for k in (wt / "kernels").glob("*_bmg.spv"):
        shutil.copy2(k, stage / "kernels" / k.name)
    (stage / "RELEASE").write_text(rid + "\n")
    version = re.search(r'VERSION = "([^"]+)"', (wt / "include/config.hpp").read_text())
    meta = {"id": rid, "commit": sha, "refs": refs, "version": version.group(1) if version else "?",
            "subject": git("log", "-1", "--format=%s", sha), "built": now_iso(), "fingerprint": fingerprint(stage)}
    save_json(stage / "meta.json", meta)
    if rel.exists():
        if release_meta(rel).get("fingerprint") == meta["fingerprint"]:
            shutil.rmtree(stage)               # identical: keep the (possibly live) dir
            return wt, rel, checks
        if rel.resolve() in (active_release(), link_target("previous")):
            raise RuntimeError(f"release {rid} is in use and would change")
        shutil.rmtree(rel)
    os.replace(stage, rel)
    return wt, rel, checks


# ------------------------------------------------------------- GPU stages --
def isolated_env(tmp, base_env=None):
    """Environment for CI GPU runs: production's GB_* settings (no
    notifications), statistics / ridge / records / live files under tmp."""
    tmp.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in NOTIFY_ENV}
    env.update({k: v for k, v in (base_env or {}).items() if k not in NOTIFY_ENV})
    env.update({"GB_STATS_STATE": str(tmp / "stats.state"), "GB_STATS_LOG": str(tmp / "stats.log"),
                "GB_RIDGE_FILE": str(tmp / "ridges.csv"), "GB_RECORDS_FILE": str(tmp / "records.txt"),
                "GB_STATS_LIVE": str(tmp / "live.json"), "GB_RECORD_HOOK": ""})
    return env


def run_bench(active, cand, run_dir, seg):
    """bench.py A/B, active vs candidate, interleaved. -> per-window paired GPU-time ratios."""
    c = cfg()
    out = run_dir / "bench.csv"
    cmd = [sys.executable, BENCH, "--windows", c["bench_windows"], "--runs", str(c["bench_runs"]),
           "--warmup", "1", "--out", out, "--segment", str(seg), "--split", "0",
           "--arm", f"active:@{active / 'goldbach_l0'}@{active / 'kernels/goldbach_bmg.spv'}",
           "--arm", f"cand:@{cand / 'goldbach_l0'}@{cand / 'kernels/goldbach_bmg.spv'}"]
    rc, _ = sh(cmd, cwd=ROOT, env=isolated_env(run_dir / "bench-files"), logf=run_dir / "bench.log", timeout=3600)
    res = {"rc": rc, "windows": {}}
    rows = list(csv.DictReader(open(out))) if out.exists() else []
    for w in c["bench_windows"].split(","):
        by = {}
        for r in rows:
            if r.get("window") == w and r.get("status") == "pass":
                by.setdefault(r["label"], {})[r["run"]] = r
        common = sorted(set(by.get("active", {})) & set(by.get("cand", {})))
        if not common:
            res["windows"][w] = {"ok": False}
            continue
        lr = [math.log(float(by["active"][k]["gpu_s"]) / float(by["cand"][k]["gpu_s"])) for k in common]
        g = math.exp(st.mean(lr))
        hw = 2.776 * st.stdev(lr) / len(lr) ** 0.5 if len(lr) > 1 else 0.0
        med = lambda arm: round(st.median(float(by[arm][k]["evens_per_sec"]) for k in common) / 1e9, 3)
        res["windows"][w] = {"ok": True, "n": len(common), "active_gevs": med("active"), "cand_gevs": med("cand"),
                             "speed_ratio": round(g, 5), "ci_pct": round(100 * (math.exp(hw) - 1), 3),
                             "gpu_slowdown_pct": round(100 * (1 / g - 1), 3)}
    return res


def run_slice(rel, a, b, seg, run_dir, base_env, timeout, name="slice"):
    """One goldbach_l0 run of [a, b] with production's arguments + env and
    isolated files. -> {rc, wall_s, csv, line, stats_line, live}."""
    tmp = run_dir / f"{name}-files"
    shutil.rmtree(tmp, ignore_errors=True)
    env = isolated_env(tmp, base_env)
    for k, v in PROD_ENV.items():                        # as run_range.sh
        env.setdefault(k, v)
    cmd = [rel / "goldbach_l0", "--start", a, "--limit", b, "--segment", seg, "--split", "0",
           "--spv", rel / "kernels/goldbach_bmg.spv", "--csv"]
    t0 = time.monotonic()
    with open(run_dir / f"{name}.out", "w") as fo, open(run_dir / f"{name}.err", "w") as fe:
        fo.write("$ " + " ".join(map(str, cmd)) + "\n")
        fo.flush()
        p = subprocess.Popen([str(x) for x in cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=fe, text=True, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            killpg(p)
            out, _ = p.communicate()
            rc = 124
        except BaseException:
            killpg(p)
            raise
        fo.write(out)
    line = next((ln for ln in out.splitlines() if "result=" in ln), "")
    return {"rc": rc, "wall_s": round(time.monotonic() - t0, 1), "csv": kv(line), "line": line,
            "stats_line": stats_line(tmp / "stats.log", a, b)}


def ref_key(fp, a, b, seg, env):
    """Cache key of a reference window result: code + window + every GB_*
    setting that can change the outcome."""
    rel_env = {k: v for k, v in sorted(env.items())
               if k.startswith("GB_") and k not in NOTIFY_ENV and not k.startswith("GB_RANGE_")}
    for k, v in PROD_ENV.items():
        rel_env.setdefault(k, v)
    rel_env = dict(sorted(rel_env.items()))
    return hashlib.sha256(json.dumps([fp, int(a), int(b), str(seg), rel_env]).encode()).hexdigest()[:20]


def compare(res, ref, prime_count=None):
    """Checks of a CI window result against a reference ({kv, stats_line})."""
    checks = []

    def chk(name, ok, detail, level_fail="fail"):
        checks.append({"name": name, "level": "ok" if ok else level_fail, "detail": detail})

    s, r = res["csv"], ref["kv"]
    chk("slice_exit", res["rc"] == 0 and s.get("result") == "PASS", f"rc={res['rc']} result={s.get('result')}")
    for k in EXACT_CSV:
        if k in r:
            chk(f"exact_{k}", s.get(k) == r.get(k), f"cand {s.get(k)} vs ref {r.get(k)}")
    if ref.get("stats_line") and res["stats_line"]:
        a_, b_ = kv(ref["stats_line"]), kv(res["stats_line"])
        # a field only the candidate writes is new statistics (e.g. the prime
        # gaps of v2.4): nothing to compare yet; a field it drops is a failure
        new = sorted(set(b_) - set(a_))
        diff = sorted(k for k in set(a_) | (set(b_) - STATS_NEW_OK) if k not in STATS_STATEFUL and a_.get(k) != b_.get(k))
        chk("exact_stats_line", not diff, ("identical (all fields but version/new_first)" if not diff
            else f"differs in {diff}") + (f"; new fields {new}" if new else ""))
        if "gaps" in b_:
            # every counted prime has exactly one gap (q - prev(q))
            chk("gaps_invariant", b_.get("gaps") == b_.get("primes") and b_.get("gsmom", "").count(",") == 3,
                f"gaps {b_.get('gaps')} vs primes {b_.get('primes')}")
    else:
        chk("exact_stats_line", False, f"missing stats line (ref {bool(ref.get('stats_line'))}, "
                                       f"cand {bool(res['stats_line'])})")
    if prime_count is None:
        chk("primecount", False, "primecount not available", level_fail="warn")
    else:
        chk("primecount", str(prime_count) == s.get("primes"), f"GPU {s.get('primes')} vs primecount {prime_count}")
    return checks


def ci_figures(run_dir, slices):
    """Waterfall figures (active vs candidate, per window) from this run's own
    data: bench.csv + the CI window runs. Best effort."""
    out = run_dir / "bench.csv"
    figs = []
    try:
        if not out.exists():
            return figs
        hdr = next(csv.reader(open(out)))
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
            for label, res in slices:
                w.writerow({**{k: res["csv"].get(k, "") for k in hdr}, "label": label, "window": "ci-window",
                            "status": "pass" if res["rc"] == 0 else "fail", "run": 1, "wall_s": res["wall_s"],
                            "session": run_dir.name, "detail": f"ci window label={label}"})
        for win in cfg()["bench_windows"].split(",") + (["ci-window"] if slices else []):
            png = run_dir / f"waterfall-{win}.png"
            rc, _ = sh([plot_python(), WATERFALL, out, "--window", win, "--group-by", "label",
                        "--label", "active,cand", "--per", "1e10", "-o", png], logf=run_dir / "figures.log", timeout=300)
            if rc == 0 and png.exists():
                figs.append(png.name)
    except Exception as e:
        log(f"figures: {type(e).__name__}: {e}")
    return figs


# --------------------------------------------------------- benchmark session --
PHASES = ["init_s", "sieve_s", "large_s", "check_s", "sync_s", "gpu_s"]


def history_build(label, commit, logf):
    """Build an old commit once (cached in ci-state/bench-builds/)."""
    d = STATE / "bench-builds" / label
    if (d / "build/goldbach_l0").exists():
        return d
    shutil.rmtree(d, ignore_errors=True)
    git("worktree", "prune")
    sh(["git", "-C", ROOT, "worktree", "add", "--detach", d, commit], logf=logf, check=True)
    (d / "tools").unlink(missing_ok=True)
    sh(["nice", "-n", "15", "cmake", "-S", d, "-B", d / "build", "-DCMAKE_BUILD_TYPE=Release"], logf=logf, check=True)
    sh(["nice", "-n", "15", "cmake", "--build", d / "build", "-j", str(cfg()["build_jobs"]), "--target", "goldbach_l0"],
       logf=logf, check=True, timeout=3600)
    return d


def bench_analysis(rows, cur, windows):
    """Markdown: medians per arm and window, and for every other arm the
    per-phase difference that explains a faster or slower GPU time."""
    L = []
    for w in windows:
        by = {}
        for r in rows:
            if r.get("window") == w and r.get("status") == "pass":
                by.setdefault(r["label"], []).append(r)
        if cur not in by:
            continue
        med = {a: {k: st.median(float(x[k]) for x in rs if x.get(k)) for k in PHASES + ["evens_per_sec"]}
               for a, rs in by.items()}
        c = med[cur]
        L += [f"### {w}", "", "| arm | G evens/s | GPU ms | init | sieve | large | check | sync | GPU time vs current |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a, m in sorted(med.items(), key=lambda x: -x[1]["evens_per_sec"]):
            L.append(f"| {a} | {m['evens_per_sec'] / 1e9:.3f} | {1e3 * m['gpu_s']:.1f} | "
                     + " | ".join(f"{1e3 * m[k]:.1f}" for k in PHASES[:-1])
                     + f" | {100 * (m['gpu_s'] / c['gpu_s'] - 1):+.2f}% |")
        L.append("")
        for a, m in med.items():
            if a == cur:
                continue
            dg = c["gpu_s"] - m["gpu_s"]
            if abs(dg) / m["gpu_s"] < 0.002:
                L.append(f"- {cur} vs {a}: no meaningful difference ({100 * dg / m['gpu_s']:+.2f}% GPU time)")
                continue
            parts = sorted(((c[k] - m[k], k) for k in ("sieve_s", "large_s", "check_s", "sync_s")),
                           key=lambda x: -abs(x[0]))
            L.append(f"- {cur} is {'SLOWER' if dg > 0 else 'faster'} than {a} by {100 * dg / m['gpu_s']:+.2f}% "
                     f"GPU time ({1e3 * dg:+.1f} ms): "
                     + ", ".join(f"{k[:-2]} {1e3 * v:+.1f} ms" for v, k in parts[:3]))
        L.append("")
    return L


def benchmark_session(out_dir, profile="full", pause=True, state=None):
    """bench.py session with production paused: current release (+ production
    config, + GB_STATS=0), previous release, history, sparse — per profile.
    Rows -> bench/bench-results.csv; figures + benchmark.md (analysis) -> out_dir."""
    c, P = cfg(), BENCH_PROFILES[profile]
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = out_dir / "benchmark.log"
    act, prev = active_release(), link_target("previous")
    cur = "v" + release_meta(act).get("version", "?")
    arms = {"cur": (cur, {}, act), "prod": (cur + "-prod", dict(PROD_ENV), act),
            "nostats": (cur + "-nostats", {"GB_STATS": "0"}, act)}
    if prev:
        pv = "v" + release_meta(prev).get("version", "?")
        arms["prev"] = (pv + ("-prev" if pv == cur else ""), {}, prev)
    sel = [arms[k] for k in P["arms"] if k in arms]
    if P["history"]:
        for label, commit in c["bench_history"].items():
            try:
                sel.append((label, {}, history_build(label, commit, logf)))
            except Exception as e:
                log(f"benchmark: history build {label} failed: {e}")

    def arm(label, env, d):
        b = d / "goldbach_l0" if (d / "goldbach_l0").exists() else d / "build/goldbach_l0"
        e = (":" + ",".join(f"{k}={v}" for k, v in env.items())) if env else ""
        return ["--arm", f"{label}{e}:@{b}@{d / 'kernels/goldbach_bmg.spv'}"]

    csv_out = out_dir / "bench.csv"
    env = isolated_env(out_dir / "files")
    t0 = time.monotonic()
    n_a = len(sel) + (1 if P["sparse"] else 0)
    n_b = len([x for x in sel if not x[0].endswith("-nostats")])
    plan = [[w, window_span(w), P["runs"] * n_a, n_a] for w in P["windows"].split(",")] + \
           [[w.split("=")[0], window_span(w), P["big_runs"] * n_b, n_b] for w in P["big"].split(",")]
    ctx = production_paused("benchmark", state) if pause else contextlib.nullcontext({})
    with ctx, flock(GPU_LOCK, blocking=True, timeout=600):
        if state is not None:
            set_phase(state, "benchmark", bench_csv=str(csv_out), bench_plan=plan, live="")
        extra = ["--arm", f"{cur}-sparse:GB_SPARSE=1:@{act / 'goldbach_l0'}"] if P["sparse"] else []
        sh([sys.executable, BENCH, *sum((arm(*x) for x in sel), []), *extra, "--windows", P["windows"],
            "--runs", str(P["runs"]), "--warmup", "1", "--split", "0", "--out", csv_out],
           cwd=ROOT, env=env, logf=logf, timeout=4 * 3600)
        big = [x for x in sel if not x[0].endswith("-nostats")]
        sh([sys.executable, BENCH, *sum((arm(*x) for x in big), []), "--windows", P["big"], "--runs",
            str(P["big_runs"]), "--warmup", "1", "--split", "0", "--out", csv_out, "--timeout", "1800"],
           cwd=ROOT, env=env, logf=logf, timeout=6 * 3600)
    paused_s = int(time.monotonic() - t0)
    rows = list(csv.DictReader(open(csv_out))) if csv_out.exists() else []
    journal = ROOT / "bench" / "bench-results.csv"  # the project's benchmark journal
    if rows and journal.exists():
        hdr = next(csv.reader(open(journal)))
        with open(journal, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
            w.writerows(rows)
    labels = list(dict.fromkeys(r["label"] for r in rows))
    dense = ",".join(lb for lb in labels if not lb.endswith("-sparse"))
    wins = [w.split("=")[0] for w in (P["windows"] + "," + P["big"]).split(",")]
    figs = []
    for win in wins:
        png = out_dir / f"waterfall-{win}-by-version.png"
        if sh([plot_python(), WATERFALL, csv_out, "--window", win, "--group-by", "label", "--label", dense,
               "--per", "1e10", "-o", png], logf=out_dir / "figures.log", timeout=300)[0] == 0:
            figs.append(png.name)
    png = out_dir / f"waterfall-{cur}-by-window.png"
    if sh([plot_python(), WATERFALL, csv_out, "--label", cur, "--group-by", "window", "--per", "1e10",
           "-o", png], logf=out_dir / "figures.log", timeout=300)[0] == 0:
        figs.append(png.name)
    md = [f"# Benchmark session {out_dir.name} ({profile})", "",
          f"Current release `{release_meta(act).get('id')}` ({cur}); arms: {', '.join(labels)}. Interleaved "
          f"fresh-process runs (tools/bench.py); production paused {paused_s} s. Rows appended to "
          "bench/bench-results.csv; figures from this session only.", "", "## Where the time goes",
          "", *bench_analysis(rows, cur, wins), "## Figures", "", *[f"![{f}]({f})" for f in figs]]
    (out_dir / "benchmark.md").write_text("\n".join(md) + "\n")
    log(f"benchmark ({profile}): {len(rows)} runs, {len(figs)} figures -> {out_dir / 'benchmark.md'}")
    return {"dir": str(out_dir), "profile": profile, "runs": len(rows), "figures": figs, "paused_s": paused_s}


# ---------------------------------------------------------------- the run --
class Run:
    """One CI run of one commit. execute() never raises for a candidate
    failure; production is resumed on every path (production_paused)."""

    def __init__(self, sha, refs, force_gpu=False, cpu_tests=True, stack_change=None):
        self.stack_change = stack_change   # set: re-verify the active release on a changed GPU stack
        self.c = cfg()
        self.sha, self.refs, self.force_gpu, self.cpu_tests = git("rev-parse", sha), refs, force_gpu, cpu_tests
        self.id = f"{stamp()}-{self.sha[:8]}"
        self.dir = STATE / "runs" / self.id
        self.dir.mkdir(parents=True)
        self.act = active_release()
        self.ameta = release_meta(self.act)
        self.pc = prod_conf()
        self.env = self.pc.get("env", {})
        self.seg = self.env.get("GB_RANGE_SEGMENT", "467500000")
        self.rec = {"run_id": self.id, "ts_start": now_iso(), "commit": self.sha, "refs": refs,
                    "kind": "gpu-stack" if stack_change else "commit", "gpu_stack": gpu_stack(),
                    "subject": git("log", "-1", "--format=%s", self.sha),
                    "previous_active": self.ameta.get("id", ""), "status": "RUNNING", "checks": []}
        self.state = {"run_id": self.id, "sha": self.sha, "refs": refs, "subject": self.rec["subject"],
                      "started": time.time(), "prod_stopped": False}

    def check(self, name, level, detail):
        self.rec["checks"].append({"name": name, "level": level, "detail": detail})

    def gate(self, msg):
        failed = [x["name"] for x in self.rec["checks"] if x["level"] == "fail"]
        if failed:
            raise CIFail(f"{msg}: {', '.join(failed)}")

    # ---- stages
    def build(self):
        set_phase(self.state, "build")
        _, rel, checks = build_release(self.sha, self.dir, self.refs, cpu_tests=self.cpu_tests)
        self.wt = STATE / "work" / "src"
        self.rec["checks"] += checks
        self.gate("build / CPU tests failed")
        self.rel, meta = rel, release_meta(rel)
        self.rec.update(release=meta["id"], version=meta["version"], fingerprint=meta["fingerprint"])

    def deployable(self):
        """None to continue, else the final status (DIVERGED / IDENTICAL)."""
        if self.stack_change:
            self.check("gpu_stack_changed", "warn", "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in
                                                             self.stack_change.items()))
            return None
        ac = self.ameta.get("commit")
        if ac and not self.c["promote_diverged"] and not is_ancestor(ac, self.sha):
            self.check("descends_from_active", "warn", f"does not contain the active commit {ac[:12]}: "
                                                       "built and tested only; merge/rebase onto it to deploy")
            return "DIVERGED"
        if self.act and self.rec["fingerprint"] == self.ameta.get("fingerprint") and not self.force_gpu:
            # no promotion: the code is the same, and moving the active commit to this
            # (possibly unmerged) branch tip would make every sibling branch that
            # lacks it DIVERGED. The older active commit is an ancestor of this one.
            self.check("artifact_identical", "ok", f"same code fingerprint as the active release "
                                                   f"{self.ameta.get('id')}: no GPU stage needed, active "
                                                   "release unchanged")
            return "IDENTICAL"
        return None

    def gpu_suite(self):
        mode = self.c["gpu_validate"]
        if not mode:
            return
        set_phase(self.state, "gpu_validate")
        args = ["bash", self.wt / "tests/gpu_validate.sh", self.wt / "build"] + (["quick"] if mode == "quick" else [])
        rc, out = sh(args, cwd=self.wt, logf=self.dir / "gpu_validate.log", timeout=3600)
        self.check(f"gpu_validate_{mode}", "ok" if rc == 0 else "fail", (out.strip().splitlines() or ["?"])[-1])
        self.gate("GPU validation suite failed")

    def bench(self):
        if not self.act or self.stack_change:     # same code on both arms: nothing to compare
            return
        set_phase(self.state, "bench")
        b = self.rec["bench"] = run_bench(self.act, self.rel, self.dir, self.seg)
        worst = max((d.get("gpu_slowdown_pct", 999) for d in b["windows"].values()), default=999)
        ok = b["rc"] == 0 and all(d.get("ok") for d in b["windows"].values())
        lvl = "fail" if not ok or worst > self.c["bench_fail_pct"] else "warn" if worst > self.c["bench_warn_pct"] else "ok"
        self.check("bench_ab", lvl, f"worst GPU-time change {worst:+.2f}% (fail > {self.c['bench_fail_pct']}%)")
        self.gate("benchmark regression or failed benchmark runs")

    def window(self):
        """(a, b, mode) of the CI window."""
        n = int(self.c["ci_slice"] or 0)
        if n:
            a = int(self.c["ci_window_start"] or self.pc["start"])
            return a, a + n - 2, "fixed"
        rows = [r for r in read_journal(prod_paths(self.pc)[0]) if r["tag"] == "PASS" and "attempts" not in r["kv"]]
        if not rows:
            raise CIFail("no banked slice for a full-slice CI window")
        return rows[-1]["a"], rows[-1]["b"], "journal"

    def reference(self, a, b, mode, fresh=False):
        """Reference result for the window: the journal row (full-slice mode),
        else the active release's cached result, run now if missing."""
        if mode == "journal":
            row = next(r for r in reversed(read_journal(prod_paths(self.pc)[0])) if r["tag"] == "PASS" and r["a"] == a)
            return {"kv": row["kv"], "stats_line": stats_line(prod_stats_log(self.pc), a, b),
                    "source": f"journal row {row['ts']}", "release": row["kv"].get("release", "?")}
        key = ref_key(self.ameta["fingerprint"], a, b, self.seg, self.env)
        cache = STATE / "ref-cache" / f"{key}.json"
        if cache.exists() and not fresh:
            ref = load_json(cache, {})
            ref["source"] = f"cached result of {ref.get('release')} ({ref.get('created')})"
            return ref
        set_phase(self.state, "slice_reference", window=[a, b], live=str(self.dir / "reference-files/live.json"))
        log(f"CI {self.id}: reference run of [{a}, {b}] with the active release {self.act.name}")
        res = run_slice(self.act, a, b, self.seg, self.dir, self.env, self.timeout(a, b), name="reference")
        if res["rc"] != 0 or res["csv"].get("result") != "PASS":
            raise RuntimeError(f"the ACTIVE release failed the reference window (rc={res['rc']}): "
                               "GPU / environment problem, not the candidate")
        self.ref_run = res
        ref = {"kv": dict(res["csv"], release=self.act.name), "stats_line": res["stats_line"],
               "release": self.act.name, "created": now_iso(), "source": "run now (same session)"}
        if self.prime_count is not None and str(self.prime_count) == res["csv"].get("primes"):
            save_json(cache, ref)            # only references primecount agrees with
        return ref

    def timeout(self, a, b):
        return max(600, int(3 * 1720 * (b - a + 2) / 2e14))

    def ci_window(self):
        a, b, mode = self.window()
        self.ref_run = None
        try:
            self.prime_count = primes_in(a, b)
        except RuntimeError as e:
            self.prime_count = None
            log(f"CI {self.id}: {e}")
        ref = self.reference(a, b, mode)
        set_phase(self.state, "slice", window=[a, b], live=str(self.dir / "slice-files/live.json"))
        log(f"CI {self.id}: window [{a}, {b}] with the candidate (reference: {ref['source']})")
        res = run_slice(self.rel, a, b, self.seg, self.dir, self.env, self.timeout(a, b))
        checks = compare(res, ref, self.prime_count)
        bad = [x for x in checks if x["level"] == "fail"]
        if bad and not self.stack_change and "cached" in ref["source"] and res["rc"] == 0 \
                and str(self.prime_count) == res["csv"].get("primes"):
            # the candidate agrees with primecount but not with the cached
            # reference: never trust a cache over a fresh run — redo it once
            log(f"CI {self.id}: cached reference disagrees ({', '.join(x['name'] for x in bad)}); re-running it")
            ref = self.reference(a, b, mode, fresh=True)
            checks = compare(res, ref, self.prime_count)
        self.rec["checks"] += checks
        if res["rc"] == 0 and "gpu_s" in res["csv"] and "gpu_s" in ref["kv"]:
            g, gr = float(res["csv"]["gpu_s"]), float(ref["kv"]["gpu_s"])
            pct = 100 * (g / gr - 1)
            lvl = "ok" if pct <= self.c["slice_warn_pct"] else "warn" if pct <= self.c["slice_fail_pct"] else "fail"
            self.check("window_gpu_time", lvl, f"{g:.1f}s vs {gr:.1f}s reference ({pct:+.2f}%)")
        self.rec["slice"] = {"start": a, "limit": b, "mode": mode, "rc": res["rc"], "wall_s": res["wall_s"],
                             "csv": {k: v for k, v in res["csv"].items() if k != "ridge_top"},
                             "ref": {k: ref["kv"].get(k) for k in TIMING_KEYS + EXACT_CSV},
                             "ref_source": ref["source"], "primecount": self.prime_count}
        self.cand_run, self.cand_key = res, (a, b)
        set_phase(self.state, "validate")
        self.rec["figures"] = ci_figures(self.dir, ([("active", self.ref_run)] if self.ref_run else [])
                                         + [("cand", res)])
        self.gate("validation failed")

    def remember_reference(self):
        """A PASS's own window result is the next run's reference (same code)."""
        a, b = self.cand_key
        if self.prime_count is None or str(self.prime_count) != self.cand_run["csv"].get("primes"):
            return
        meta = release_meta(self.rel)
        save_json(STATE / "ref-cache" / f"{ref_key(meta['fingerprint'], a, b, self.seg, self.env)}.json",
                  {"kv": dict(self.cand_run["csv"], release=self.rel.name), "stats_line": self.cand_run["stats_line"],
                   "release": self.rel.name, "created": now_iso()})

    # ---- driver
    def execute(self):
        log(f"CI {self.id}: {self.sha[:12]} ({', '.join(self.refs)}) — {self.rec['subject']}")
        t_down = None
        try:
            self.build()
            final = self.deployable()
            if final:
                self.rec["status"] = final
                return self.finish()
            self.rec["config"] = {"segment": self.seg, "split": 0, "bucket_verify": self.env.get("GB_BUCKET_VERIFY", PROD_ENV["GB_BUCKET_VERIFY"]),
                                  "gaps": self.env.get("GB_GAPS", PROD_ENV["GB_GAPS"]),
                                  "env_keys": sorted(k for k in self.env if k not in NOTIFY_ENV),
                                  "ci_slice": self.c["ci_slice"], "bench_windows": self.c["bench_windows"]}
            with production_paused(f"CI {self.id}", self.state) as info:
                self.rec["production"] = info
                t_down = time.monotonic()
                try:
                    with flock(GPU_LOCK, blocking=True, timeout=120):
                        self.gpu_suite()
                        self.bench()
                        self.ci_window()
                except CIFail:
                    if self.stack_change:      # the running release no longer reproduces its
                        self.state["no_resume"] = True       # own results: keep production stopped
                    raise
                self.rec["status"] = "PASS"
                save_json(STATE / "gpu-stack.json", self.rec["gpu_stack"])   # verified on this stack
                if self.stack_change:
                    log(f"CI {self.id}: the active release reproduces its reference on the new GPU stack")
                else:
                    promote(self.rel)
                    self.remember_reference()
                    prof = self.c["bench_on_promote"]
                    if prof:
                        try:
                            self.rec["benchmark"] = benchmark_session(self.dir / "benchmark", prof, pause=False,
                                                                      state=self.state)
                        except Exception as e:
                            self.rec["benchmark"] = {"error": f"{type(e).__name__}: {e}"}
                            log(f"CI {self.id}: benchmark error (the PASS stands): {e}")
        except CIFail as e:
            self.rec["status"], self.rec["reason"] = "FAIL", str(e)
        except Exception as e:                  # infrastructure error: same safe path
            self.rec["status"], self.rec["reason"] = "ERROR", f"{type(e).__name__}: {e}"
        if self.state.get("no_resume"):
            pc = prod_conf()
            pc["desired"] = "stopped"
            save_prod_conf(pc)
            alert("GPU STACK CHANGE BREAKS RESULTS", f"{self.rec.get('reason')} after "
                  + ", ".join(self.stack_change) + " changed — production stopped; roll back the update "
                  "or re-validate (goldbach_ci.py production start once fixed)")
        if self.stack_change:
            if self.rec["status"] == "PASS":
                (STATE / GPU_STACK_FAILED).unlink(missing_ok=True)
            else:                               # not again on this stack (see gpu_stack_check_due)
                save_json(STATE / GPU_STACK_FAILED, self.rec["gpu_stack"])
                log(f"CI {self.id}: GPU stack check {self.rec['status']}; not retried on this stack "
                    f"(delete ci-state/{GPU_STACK_FAILED} to retry)")
        if t_down is not None:
            self.rec["downtime_s"] = int(time.monotonic() - t_down)
        return self.finish()

    def finish(self):
        rec = self.rec
        rec["ts_end"] = now_iso()
        write_report(self.dir, rec)
        ledger_append(rec)
        save_json(STATE / "state.json", {"phase": "idle", "last_run": rec["run_id"]})
        log(f"CI {rec['run_id']}: {rec['status']}" + (f" — {rec['reason']}" if rec.get("reason") else "")
            + f" — {self.dir / 'report.md'}")
        if rec["status"] in ("FAIL", "ERROR"):
            alert(f"CI {rec['status']}: {rec['commit'][:10]}",
                  f"{rec.get('reason', '')} — active release stays {release_meta(active_release()).get('id', '-')}")
        return rec


def ledger_append(rec):
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "ledger.jsonl", "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    sl = rec.get("slice", {})
    flat = {"run_id": rec["run_id"], "ts_start": rec.get("ts_start", ""), "ts_end": rec.get("ts_end", ""),
            "status": rec["status"], "reason": rec.get("reason", ""), "refs": " ".join(rec.get("refs", [])),
            "commit": rec.get("commit", ""), "version": rec.get("version", ""), "release": rec.get("release", ""),
            "previous_active": rec.get("previous_active", ""), "segment": rec.get("config", {}).get("segment", ""),
            "window_start": sl.get("start", ""), "window_limit": sl.get("limit", ""),
            "reference": sl.get("ref_source", ""), "slice_wall_s": sl.get("wall_s", ""),
            "checks_failed": ";".join(x["name"] for x in rec.get("checks", []) if x["level"] == "fail"),
            "checks_warned": ";".join(x["name"] for x in rec.get("checks", []) if x["level"] == "warn"),
            "downtime_s": rec.get("downtime_s", ""), "lost_s": rec.get("production", {}).get("lost_s", "")}
    for k in TIMING_KEYS:
        flat["slice_" + k] = sl.get("csv", {}).get(k, "")
    for w, d in rec.get("bench", {}).get("windows", {}).items():
        flat[f"bench_{w}_speed_ratio"] = d.get("speed_ratio", "")
    p = STATE / "results.csv"
    if p.exists() and next(csv.reader(open(p)), None) != RESULT_COLS:
        p.rename(p.with_name(f"results-{stamp()}.csv"))      # older layout: keep it aside
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(flat)


def ledger():
    try:
        return [json.loads(ln) for ln in open(STATE / "ledger.jsonl")]
    except OSError:
        return []


def write_report(run_dir, rec):
    icon = {"PASS": "✅", "IDENTICAL": "✅", "DIVERGED": "⚠️", "FAIL": "❌", "ERROR": "❌"}.get(rec["status"], "")
    L = [f"# {icon} {rec['status']} — `{rec.get('commit', '')[:12]}` ({', '.join(rec.get('refs', []))})", "",
         f"{rec.get('subject', '')}", ""]
    if rec.get("reason"):
        L += [f"**{rec['reason']}**", ""]
    L += ["| check | result | detail |", "|---|---|---|"]
    L += [f"| {x['name']} | {x['level'].upper()} | {x['detail']} |" for x in rec.get("checks", [])]
    pr = rec.get("production", {})
    L += ["", f"version {rec.get('version', '?')} · release `{rec.get('release', '-')}` · previous active "
              f"`{rec.get('previous_active', '-')}` · {rec.get('ts_start', '')} → {rec.get('ts_end', '')} · "
              f"production paused {rec.get('downtime_s', 0)} s"
              + (f" ({pr['mode']}, {pr['lost_s']} s of a partial slice redone)" if pr.get("lost_s") else ""), ""]
    if rec.get("slice"):
        s = rec["slice"]
        L += [f"## CI window [{s['start']}, {s['limit']}]", "", f"reference: {s.get('ref_source', '?')}; "
              f"primecount: {s.get('primecount')}", "", "| metric | candidate | reference |", "|---|---|---|"]
        L += [f"| {k} | {s['csv'].get(k, '')} | {s.get('ref', {}).get(k, '')} |" for k in TIMING_KEYS]
        L.append("")
    if rec.get("bench", {}).get("windows"):
        L += ["## bench.py A/B (interleaved, paired)", "",
              "| window | active G ev/s | candidate G ev/s | speed ratio | ±95% |", "|---|---|---|---|---|"]
        L += [f"| {w} | {d.get('active_gevs', '')} | {d.get('cand_gevs', '')} | {d.get('speed_ratio', '')} | "
              f"{d.get('ci_pct', '')}% |" for w, d in rec["bench"]["windows"].items()]
        L.append("")
    b = rec.get("benchmark", {})
    if b.get("dir"):
        L += [f"## Benchmark session ({b['profile']})", "", (Path(b["dir"]) / "benchmark.md").read_text()
              .split("## Where the time goes", 1)[-1].split("## Figures", 1)[0].strip(), ""]
    if rec.get("figures") or b.get("figures"):
        L += ["Figures (artifact): " + ", ".join(f"`{f}`" for f in rec.get("figures", []) + b.get("figures", [])), ""]
    (run_dir / "report.md").write_text("\n".join(L) + "\n")
    save_json(run_dir / "report.json", rec)


# -------------------------------------------------------- detection / queue --
def list_refs(c):
    pats = list(c["watch_include"]) + (["refs/remotes/origin/*"] if c["fetch"] else [])
    refs = []
    for ln in git("for-each-ref", "--format=%(refname) %(objectname) %(committerdate:unix)").splitlines():
        name, sha, ts = ln.split()
        if not name.endswith("/HEAD") and any(fnmatch.fnmatch(name, p) for p in pats) \
                and not any(fnmatch.fnmatch(name, p) for p in c["watch_exclude"]):
            refs.append((name, sha, int(ts)))
    return refs


def attempts(sha):
    return sum(r.get("commit") == sha and r["status"] == "INTERRUPTED" for r in ledger())


def judged(sha):
    return any(r.get("commit") == sha and r["status"] in FINAL for r in ledger())


def candidates(update_seen=True):
    """New (sha, [refs]) tips since the last poll, oldest commit first."""
    c = cfg()
    if c["fetch"]:
        subprocess.run(["git", "-C", str(ROOT), "fetch", "--quiet", "origin"], capture_output=True)
    seen = load_json(STATE / "refs.json", {})
    refs = list_refs(c)
    act = release_meta(active_release()).get("commit")
    new = {}
    for name, sha, ts in refs:
        if seen.get(name) == sha:
            continue
        if judged(sha) or attempts(sha) >= c["max_attempts"] or (act and is_ancestor(sha, act)):
            seen[name] = sha
            continue
        new.setdefault(sha, {"refs": [], "ts": ts})["refs"].append(name.replace("refs/heads/", ""))
    seen = {k: v for k, v in seen.items() if k in {n for n, _, _ in refs}}
    if update_seen:
        save_json(STATE / "refs.json", seen)
    depth = lambda s: int(git("rev-list", "--count", s))     # ancestors first on equal dates
    return sorted(((s, d["refs"]) for s, d in new.items()), key=lambda x: (new[x[0]]["ts"], depth(x[0])))


def mark_seen(sha):
    seen = load_json(STATE / "refs.json", {})
    for name, s, _ in list_refs(cfg()):
        if s == sha:
            seen[name] = sha
    save_json(STATE / "refs.json", seen)


def queue_edit(fn):
    q = load_json(STATE / "queue.json", [])
    q = fn(q)
    save_json(STATE / "queue.json", q)
    return q


# ------------------------------------------------------------- recovery ---
def recover():
    """After a crash / reboot: kill GPU jobs a dead CI left behind, record an
    interrupted run and queue it for a retry, restore production."""
    stt = load_json(STATE / "state.json", {"phase": "idle"})
    for pid in pgrep("goldbach_l0"):
        exe = " ".join(c.decode(errors="replace") for c in proc_cmdline(pid))
        if str(STATE) in exe and not find_run_range():
            log(f"recovery: killing orphaned CI GPU job {pid}")
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
    if stt.get("phase", "idle") != "idle":
        sha = stt.get("sha", "")
        log(f"recovery: CI run {stt.get('run_id')} was interrupted in phase {stt['phase']}")
        ledger_append({"run_id": stt.get("run_id", "?"), "ts_end": now_iso(), "commit": sha,
                       "refs": stt.get("refs", []), "status": "INTERRUPTED", "checks": [],
                       "reason": f"CI process died in phase {stt['phase']} (crash / reboot / kill)"})
        for d in (STATE / "releases").glob("*.staging"):
            shutil.rmtree(d, ignore_errors=True)
        save_json(STATE / "state.json", {"phase": "idle", "recovered": stt.get("run_id")})
        if sha and attempts(sha) < cfg()["max_attempts"] and not judged(sha):
            queue_edit(lambda q: [x for x in q if x["sha"] != sha] +
                       [{"sha": sha, "refs": stt.get("refs", []), "reason": "retry after interruption",
                         "added": now_iso()}])
            log(f"recovery: {sha[:12]} queued for a retry")
        else:
            alert("CI run interrupted", f"{sha[:10]} in phase {stt['phase']}; not retried again")
    pc = prod_conf()
    if pc.get("desired") == "running" and not find_run_range() and not pgrep("goldbach_l0"):
        last = last_run_range_event(prod_paths(pc)[1])
        if re.search(r"STOPPED|FATAL", last):
            log(f"recovery: production is down after a failure, NOT restarting: {last}")
            alert("run_range is stopped", last[:300])
        else:
            log("recovery: production should be running; resuming it")
            try:
                start_production()
            except Exception as e:
                log(f"recovery: resume failed: {e}")


# --------------------------------------------------------- count check ----
def count_check(journal=None, only_new=True, limit_blocks=None, stop_on_mismatch=False, partial=False):
    """Banked slices' GPU prime counts vs primecount, in blocks of consecutive
    slices covering >= count_block integers (one primecount call per block,
    whatever the slice size); a block that disagrees is re-checked slice by
    slice to find the slice. Results: count-check.jsonl (one row per block,
    plus one per mismatching slice). `partial` also checks a trailing block
    shorter than count_block."""
    c, pc = cfg(), prod_conf()
    j = Path(journal) if journal else prod_paths(pc)[0]
    # resolved = inside a block that matched, or already judged on its own
    # (a known mismatch is handled by the recheck queue and alerted once), or
    # re-verified by `recheck` (RECHECK row in the journal) -- by THIS
    # journal's runs only: another journal over the same integers (a
    # backfill) is a separate GPU run and is checked on its own. Rows before
    # the "journal" field count for slices of the releases they name.
    covered, judged_alone, rechecked = [], {}, set()
    with contextlib.suppress(OSError):
        for r in map(json.loads, open(STATE / "count-check.jsonl")):
            if "journal" in r and r["journal"] != j.name:
                continue
            rels = None if "journal" in r else set(r.get("releases") or [r.get("release", "")])
            if r.get("ok"):
                covered.append((r["a"], r["b"], rels))
            if r.get("slices", 1) == 1:
                judged_alone.setdefault((r["a"], r["b"]), []).append(rels)
    with contextlib.suppress(OSError):
        rechecked = {(int(t[1]), int(t[2])) for t in (ln.split() for ln in open(j)) if t[:1] == ["RECHECK"]}

    def resolved(r):
        mine = lambda rels: rels is None or r["kv"].get("release", "") in rels
        return ((r["a"], r["b"]) in rechecked or any(map(mine, judged_alone.get((r["a"], r["b"]), [])))
                or any(a <= r["a"] and r["b"] <= b and mine(rels) for a, b, rels in covered))
    rows = sorted((r for r in read_journal(j) if r["tag"] == "PASS" and "primes" in r["kv"]
                   and not (only_new and resolved(r))), key=lambda r: r["a"])
    blocks, cur = [], []
    for r in rows:
        if cur and r["a"] != cur[-1]["b"] + 2:
            blocks.append(cur)             # a gap (an unbanked or already checked slice) ends a block
            cur = []
        cur.append(r)
        if cur[-1]["b"] - cur[0]["a"] + 2 >= c["count_block"]:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    blocks = [bl for bl in blocks if partial or bl[-1]["b"] - bl[0]["a"] + 2 >= c["count_block"]
              or bl is not blocks[-1]][:limit_blocks]

    def record(out):
        with open(STATE / "count-check.jsonl", "a") as f:
            f.write(json.dumps(out) + "\n")

    bad = []
    for bl in blocks:
        t0 = time.monotonic()
        a, b = bl[0]["a"], bl[-1]["b"]
        gpu = sum(int(r["kv"]["primes"]) for r in bl)
        # the odd integer between two consecutive slices ([a, b], [b + 2, ...]) is in
        # no slice but in [a, b] of the block: a prime there is not a GPU miss
        between = sum(1 for r in bl[:-1] if is_prime(r["b"] + 1))
        ref = primes_in(a, b) - between
        out = {"a": a, "b": b, "slices": len(bl), "gpu": gpu, "primecount": ref, "between": between, "ok": gpu == ref,
               "releases": sorted({r["kv"].get("release", "") for r in bl}), "journal": j.name, "checked": now_iso(),
               "secs": round(time.monotonic() - t0, 1)}
        record(out)
        print(f"[{a}, {b}] {len(bl)} slice(s): GPU {gpu} primecount {ref} {'OK' if out['ok'] else 'MISMATCH'} "
              f"({out['secs']}s)", flush=True)
        if out["ok"]:
            continue
        for r in bl:                       # localise: slice by slice
            ref_r = primes_in(r["a"], r["b"])
            if str(ref_r) == r["kv"]["primes"]:
                continue
            o = {"a": r["a"], "b": r["b"], "slices": 1, "gpu": int(r["kv"]["primes"]), "primecount": ref_r,
                 "ok": False, "release": r["kv"].get("release", ""), "journal": j.name, "checked": now_iso()}
            record(o)
            bad.append(o)
            log(f"PRIME COUNT MISMATCH in banked slice [{r['a']}, {r['b']}]: GPU {o['gpu']} vs primecount {ref_r}")
            alert("PRIME COUNT MISMATCH", f"slice [{r['a']}, {r['b']}]: GPU {o['gpu']} vs primecount {ref_r} — "
                                          "this PASS is not trustworthy; `goldbach_ci.py recheck` it")
            with open(j.with_suffix(".recheck"), "a") as f:
                f.write(f"{r['a']} {r['b']} {now_iso()} count_mismatch gpu={o['gpu']} primecount={ref_r}\n")
    if bad and stop_on_mismatch and find_run_range():
        stop_production(reason="prime-count mismatch")
        pc = prod_conf()                  # stop_production may have adopted / saved it
        pc["desired"] = "stopped"
        save_prod_conf(pc)
    return len(blocks), bad


# --------------------------------------------------------------- recheck --
def recheck(a=None, release=None):
    """Re-run banked slice(s) — default: the .recheck queue — and compare them
    with their journal rows and primecount; a RECHECK row records the outcome."""
    pc = prod_conf()
    j = prod_paths(pc)[0]
    rows = {(r["a"], r["b"]): r for r in read_journal(j) if r["tag"] == "PASS"}
    if a is not None:
        queue = [k for k in rows if k[0] == int(a)]
    else:
        done = set()
        with contextlib.suppress(OSError):
            done = {tuple(int(x) for x in ln.split()[:2]) for ln in open(STATE / "rechecked.txt")}
        qf = j.with_suffix(".recheck")
        queue = list(dict.fromkeys(k for k in (tuple(int(x) for x in ln.split()[:2])
                                                for ln in (open(qf) if qf.exists() else []))
                                   if k in rows and k not in done))
    if not queue:
        print("nothing to recheck")
        return []
    rel = release_dir(release) if release else active_release()
    seg = pc.get("env", {}).get("GB_RANGE_SEGMENT", "467500000")
    results = []
    with production_paused("recheck"), flock(GPU_LOCK, blocking=True, timeout=120):
        for k in queue:
            ref = rows[k]
            run_dir = STATE / "runs" / f"{stamp()}-recheck-{k[0]}"
            run_dir.mkdir(parents=True)
            log(f"recheck [{k[0]}, {k[1]}] with release {rel.name}")
            # the gap walk as the slice was banked (rows before gap_mode= walked every segment)
            env = dict(pc.get("env", {}), GB_GAPS=ref["kv"].get("gap_mode", "exact"))
            res = run_slice(rel, k[0], k[1], seg, run_dir, env,
                            max(600, int(3 * float(ref["kv"].get("total_s", 1800)))))
            try:
                count = primes_in(*k)
            except RuntimeError:
                count = None
            checks = compare(res, {"kv": ref["kv"], "stats_line": stats_line(prod_stats_log(pc), *k)}, count)
            ok = not any(x["level"] == "fail" for x in checks)
            save_json(run_dir / "report.json", {"slice": k, "checks": checks, "ok": ok, "csv": res["csv"]})
            with open(STATE / "rechecked.txt", "a") as f:
                f.write(f"{k[0]} {k[1]} {now_iso()} {'CONFIRMED' if ok else 'MISMATCH'} {rel.name}\n")
            with open(j, "a") as f:        # non-PASS row: ignored by the resume chain and dashboards
                f.write(f"RECHECK {k[0]} {k[1]} {now_iso()} outcome={'CONFIRMED' if ok else 'DIFFERS'} "
                        f"release={rel.name} primes={res['csv'].get('primes')} twins={res['csv'].get('twins')} "
                        f"result={res['csv'].get('result')} primecount={count}\n")
            log(f"recheck [{k[0]}, {k[1]}]: {'CONFIRMED' if ok else 'MISMATCH'}")
            if not ok:
                alert("RECHECK MISMATCH", f"slice [{k[0]}, {k[1]}] re-run differs: "
                      + ", ".join(x["name"] for x in checks if x["level"] == "fail"))
            results.append((k, ok))
    return results


# ------------------------------------------------------------ progress ----
BENCH_PRESETS = {"bench": 2e9, "steady": 2e10, "end": 2e10, "long": 1e11, "long-end": 1e11}


def window_span(spec):
    """Integers of a bench.py window spec (preset or NAME=START:LIMIT)."""
    if "=" in spec:
        a, b = spec.split("=", 1)[1].split(":")
        return int(b) - int(a)
    return BENCH_PRESETS.get(spec, 1e11)


_energy = {}


def gpu_stats():
    """'GPU 2850 MHz 71 °C 152 W' for the Intel card (sysfs; power from the
    energy counter between two calls)."""
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        try:
            if (card / "device/vendor").read_text().strip() != "0x8086":
                continue
        except OSError:
            continue
        parts = []
        with contextlib.suppress(OSError, ValueError):
            parts.append(f"{int((card / 'device/tile0/gt0/freq0/act_freq').read_text())} MHz")
        hw = next(iter(sorted((card / "device/hwmon").glob("hwmon*"))), None)
        if hw:
            for lab in hw.glob("temp*_label"):
                if lab.read_text().strip() == "pkg":
                    with contextlib.suppress(OSError, ValueError):
                        parts.append(f"{int(lab.with_name(lab.name.replace('_label', '_input')).read_text()) // 1000} °C")
            with contextlib.suppress(OSError, ValueError):
                e, t = int((hw / "energy1_input").read_text()), time.monotonic()
                if "e" in _energy and t > _energy["t"]:
                    parts.append(f"{(e - _energy['e']) / 1e6 / (t - _energy['t']):.0f} W")
                _energy.update(e=e, t=t)
        return "GPU " + " ".join(parts) if parts else ""
    return ""


def bench_progress(stt):
    """Progress of a benchmark session from its bench.csv and plan: % of the
    planned work (weighted by each window's run time), runs done, the
    latest run (arm, window, speed) and an ETA."""
    plan = stt.get("bench_plan") or []
    try:
        rows = list(csv.DictReader(open(stt["bench_csv"])))
    except (OSError, KeyError):
        rows = []
    total = sum(p[2] for p in plan)
    if not plan:
        return ""
    done_w = {}
    for r in rows:
        done_w.setdefault(r.get("window"), []).append(r)
    cost = {}
    for w, span, _, _ in plan:      # seconds per run: measured, else ~58 G evens/s + 0.3 s start
        ws = [float(r["wall_s"]) for r in done_w.get(w, []) if r.get("wall_s")]
        cost[w] = st.median(ws) if ws else span / 2 / 58e9 + 0.3
    # warm-up runs (one per arm and window) come first and are not in the csv
    tot = sum((n + k) * cost[w] for w, _, n, k in plan)
    done = sum((len(done_w.get(w, [])) + (k if done_w.get(w) else 0)) * cost[w] for w, _, n, k in plan)
    el = time.time() - stt.get("phase_since", time.time())
    fr = min(done / tot, 1.0) if tot else 0
    eta = el * (1 - fr) / fr if fr > 0.02 else (tot - done)
    line = f"{100 * fr:.0f}% ({len(rows)}/{total} runs)"
    if rows:
        r = rows[-1]
        with contextlib.suppress(ValueError, KeyError):
            line += f" | now {r['label']} on {r['window']}: {float(r['evens_per_sec']) / 1e9:.2f} G ev/s"
    return line + f" | ETA {hms(eta)}"


def hms(s):
    s = int(max(s or 0, 0))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def progress_line():
    """One line: what the CI is doing right now (status, Actions heartbeat)."""
    stt = load_json(STATE / "state.json", {})
    ph = stt.get("phase", "idle")
    if ph == "idle":
        return "idle"
    line = f"{stt.get('sha', '')[:10]} {ph} for {hms(time.time() - stt.get('phase_since', time.time()))}"
    if ph == "stop_production":
        lv = prod_live(prod_conf())
        if lv and lv.get("elapsed_s"):
            fr = lv["evens_done"] / max(lv["evens_total"], 1)
            eta = lv["elapsed_s"] * (1 - fr) / fr if fr else 0
            line += f": production slice {100 * fr:.1f}% done, boundary in ~{hms(eta)}"
    elif ph in ("slice", "slice_reference") and stt.get("live"):
        lv = load_json(stt["live"], {})
        if lv.get("evens_total") and lv.get("elapsed_s"):
            fr = lv["evens_done"] / lv["evens_total"]
            r = lv["evens_done"] / lv["elapsed_s"]
            line += (f": {'reference' if ph == 'slice_reference' else 'candidate'} window {100 * fr:.1f}% "
                     f"| {r / 1e9:.2f} G ev/s | ETA {hms((lv['evens_total'] - lv['evens_done']) / r if r else 0)}")
    elif ph == "benchmark":
        line += ": " + bench_progress(stt)
    if ph in ("slice", "slice_reference", "benchmark", "gpu_validate", "bench"):
        g = gpu_stats()
        line += f" | {g}" if g else ""
    return line


@contextlib.contextmanager
def heartbeat(every=60):
    """Print progress_line() every `every` s (keeps an Actions log alive)."""
    stop = threading.Event()

    def beat():
        while not stop.wait(every):
            print(f"{now_iso()} … {progress_line()}", flush=True)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()


def annotate(rec):
    """GitHub Actions annotations for failed / warned checks."""
    if not ON_ACTIONS:
        return
    for x in rec.get("checks", []):
        if x["level"] in ("fail", "warn"):
            kind = "error" if x["level"] == "fail" else "warning"
            print(f"::{kind} title=CI {x['name']}::{x['detail']}", flush=True)
    print(f"::notice title=CI {rec['status']}::{rec.get('reason') or rec.get('subject', '')}", flush=True)


# --------------------------------------------------------------- commands --
def cmd_bootstrap(a):
    """First known-good release from HEAD (or --rev); if ./build/goldbach_l0
    exists it must have the same code fingerprint."""
    rev = git("rev-parse", a.rev or "HEAD")
    run_dir = STATE / "runs" / f"{stamp()}-bootstrap"
    run_dir.mkdir(parents=True)
    with flock(STATE / "ci.lock"):
        _, rel, checks = build_release(rev, run_dir, ["bootstrap"])
        for x in checks:
            print(f"  {x['name']:16s} {x['level']:5s} {x['detail']}")
        if not rel:
            sys.exit(f"bootstrap build failed; see {run_dir}")
        prod_bin = ROOT / "build" / "goldbach_l0"
        if prod_bin.exists() and not a.no_verify:
            with tempfile.TemporaryDirectory() as td:
                t = Path(td)
                (t / "kernels").mkdir()
                shutil.copy2(prod_bin, t / "goldbach_l0")
                for k in (ROOT / "kernels").glob("*_bmg.spv"):
                    shutil.copy2(k, t / "kernels" / k.name)
                same = fingerprint(t) == release_meta(rel)["fingerprint"]
            print(f"  ./build/goldbach_l0 + kernels/ fingerprint {'==' if same else '!='} release {rel.name}")
            if not same:
                sys.exit("the production binary is NOT this commit's code: --rev the commit it was built "
                         "from, or --no-verify")
        promote(rel)
        if find_run_range() and not prod_conf():
            adopt_production()
        save_json(STATE / "refs.json", {name: sha for name, sha, _ in list_refs(cfg())})   # baseline
        ledger_append({"run_id": run_dir.name, "ts_start": now_iso(), "ts_end": now_iso(), "commit": rev,
                       "refs": ["bootstrap"], "status": "BOOTSTRAP", "release": rel.name,
                       "version": release_meta(rel)["version"], "checks": checks})
        print(f"active release: {rel.name}  ({release_meta(rel)['subject']})")


def cmd_status(a):
    am, pm = release_meta(active_release()), release_meta(link_target("previous"))
    print(f"active    {am.get('id', '-')}  v{am.get('version', '?')}  {am.get('subject', '')}")
    print(f"previous  {pm.get('id', '-')}  v{pm.get('version', '?')}  {pm.get('subject', '')}")
    ps, pc = production_status(), prod_conf()
    procs = ps["procs"]
    print(f"production {'RUNNING' if procs else 'stopped'} (desired {ps['desired']})"
          + (f", pid {procs[0]['pid']}" + (f", unit {procs[0]['unit']}" if procs[0]["unit"] else ", not CI-managed")
             if procs else ""))
    if pc:
        print(f"  {ps.get('banked')} slices banked, resume point {ps.get('next')}; "
              f"last event: {last_run_range_event(prod_paths(pc)[1])[26:150]}")
    print(f"CI        {progress_line()}")
    ch = {k: v for k, v in gpu_stack().items() if load_json(STATE / "gpu-stack.json", {}).get(k) != v}
    failed = ch and load_json(STATE / GPU_STACK_FAILED, None) == gpu_stack()
    print("GPU stack " + ("verified" if not ch else ("CHANGED, verification FAILED (not retried): " if failed
                                                     else "CHANGED, verification pending: ") + ", ".join(ch)))
    q = load_json(STATE / "queue.json", [])
    if q:
        print("queue     " + ", ".join(f"{x['sha'][:10]} ({x['reason']})" for x in q))
    rows = ledger()[-a.n:]
    print(f"\nlast {len(rows)} runs:")
    for r in rows:
        sl = r.get("slice", {}).get("csv", {})
        print(f"  {r['run_id']:30s} {r['status']:11s} {r.get('commit', '')[:10]:10s} "
              f"{','.join(r.get('refs', []))[:20]:20s} {('gpu_s=' + sl['gpu_s']) if sl.get('gpu_s') else '':16s} "
              f"{r.get('reason', '')[:70]}")
    if cfg()["watch_include"]:
        cands = candidates(update_seen=False)
        print(f"\npending branch tips: {', '.join(s[:10] + '(' + ','.join(r) + ')' for s, r in cands) or 'none'}")


def poll_once():
    c = cfg()
    if c["count_check"] and primecount_bin() and prod_conf():
        try:
            count_check(stop_on_mismatch=c["count_check_stop"], limit_blocks=2)
        except Exception as e:
            log(f"count-check error: {e}")
    ran = 0
    change = gpu_stack_check_due() if active_release() else {}
    if change:
        log("GPU stack changed since the active release was verified: "
            + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in change.items()))
        Run(release_meta(active_release())["commit"], ["gpu-stack-check"], cpu_tests=False,
            stack_change=change).execute()
        ran += 1
    work = [(x["sha"], x["refs"]) for x in load_json(STATE / "queue.json", [])] + candidates(update_seen=False)
    for sha, refs in work:
        queue_edit(lambda q: [x for x in q if x["sha"] != sha])
        if judged(sha):
            continue
        Run(sha, refs).execute()
        mark_seen(sha)
        ran += 1
    if c["auto_recheck"]:
        recheck()
    return ran


def cmd_poll(a):
    with flock(STATE / "ci.lock"):
        poll_once()


def cmd_daemon(a):
    # one daemon; ci.lock (shared with `run`) is held only while a poll works
    with flock(STATE / "daemon.lock"):
        log(f"daemon start (pid {os.getpid()}, poll {cfg()['poll_s']}s)")
        with flock(STATE / "ci.lock", blocking=True, timeout=6 * 3600):
            recover()
        while True:
            try:
                lock = flock(STATE / "ci.lock")
                lock.__enter__()
            except RuntimeError:          # a CI run (Actions job, command) is busy: next poll
                time.sleep(cfg()["poll_s"])
                continue
            try:
                poll_once()
            except Exception as e:
                log(f"daemon: poll {type(e).__name__}: {e}")
            finally:
                lock.__exit__(None, None, None)
            time.sleep(cfg()["poll_s"])


def cmd_run(a):
    if a.fetch:     # e.g. a GitHub Actions job: the commit may only exist on origin
        subprocess.run(["git", "-C", str(ROOT), "fetch", "--quiet", "origin", a.rev], check=False)
    sha = git("rev-parse", a.rev + "^{commit}")
    with heartbeat(30):
        log(f"run {sha[:12]}: waiting for the CI lock ({progress_line()})")
        with flock(STATE / "ci.lock", blocking=True, timeout=6 * 3600):
            if not active_release():
                sys.exit("no active release: run bootstrap first")
            refs = [a.ref] if a.ref else [ln.replace("refs/heads/", "") for ln in
                                          git("for-each-ref", "--points-at", sha, "--format=%(refname)").splitlines()] or [a.rev]
            queue_edit(lambda q: [x for x in q if x["sha"] != sha])
            prior = None if (a.rerun or a.force_gpu) else next(
                (r for r in reversed(ledger()) if r.get("commit") == sha and r["status"] in FINAL), None)
            if prior:
                log(f"run {sha[:12]}: already judged in {prior['run_id']} ({prior['status']}); reporting it "
                    "(--rerun repeats it)")
                rec = prior
            else:
                rec = Run(sha, refs, force_gpu=a.force_gpu, cpu_tests=not a.skip_cpu_tests).execute()
            mark_seen(sha)
    run_dir = STATE / "runs" / rec["run_id"]
    annotate(rec)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write((run_dir / "report.md").read_text())
    if a.artifacts:
        shutil.copytree(run_dir, a.artifacts, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*-files", "files"))
    sys.exit(0 if rec["status"] in ("PASS", "IDENTICAL", "DIVERGED") else 1)


def cmd_production(a):
    if a.action == "status":
        print(json.dumps(production_status(), indent=1, default=str))
    elif a.action == "adopt":
        adopt_production()
    else:
        with flock(STATE / "ci.lock"):
            if a.action == "stop":
                stop_production(now=a.now, reason="manual")
                pc = prod_conf()
                pc["desired"] = "stopped"
                save_prod_conf(pc)
            else:
                start_production()


def cmd_rollback(a):
    prev = link_target("previous")
    if not prev:
        sys.exit("no previous release")
    with flock(STATE / "ci.lock"):
        running = bool(find_run_range())
        if running:
            stop_production(now=a.now, reason="rollback")
        cur = active_release()
        set_link("active", prev)
        set_link("previous", cur)
        log(f"rollback: active = {prev.name}, previous = {cur.name}")
        ledger_append({"run_id": f"{stamp()}-rollback", "ts_start": now_iso(), "ts_end": now_iso(),
                       "status": "ROLLBACK", "release": prev.name, "previous_active": cur.name,
                       "commit": release_meta(prev).get("commit", ""), "checks": []})
        if running:
            start_production()


def cmd_count_check(a):
    n, bad = count_check(journal=a.journal, only_new=not a.all, limit_blocks=a.max, partial=True)
    print(f"{n} block(s) checked, {len(bad)} mismatching slice(s)")
    sys.exit(1 if bad else 0)


def cmd_recheck(a):
    with flock(STATE / "ci.lock"):
        res = recheck(a.start, a.release)
    sys.exit(0 if all(ok for _, ok in res) else 1)


def cmd_benchmark(a):
    with heartbeat(30), flock(STATE / "ci.lock", blocking=True, timeout=6 * 3600):
        d = STATE / "runs" / f"{stamp()}-benchmark-{a.profile}"
        state = {"run_id": d.name, "sha": release_meta(active_release()).get("commit", ""), "refs": ["benchmark"],
                 "subject": f"benchmark session ({a.profile})", "started": time.time()}
        try:
            benchmark_session(d, a.profile, pause=True, state=state)
        finally:
            save_json(STATE / "state.json", {"phase": "idle", "last_run": d.name})
    md = (d / "benchmark.md").read_text()
    print(md)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(md)
    if a.artifacts:
        shutil.copytree(d, a.artifacts, dirs_exist_ok=True, ignore=shutil.ignore_patterns("files"))


def cmd_install_service(a):
    d = Path.home() / ".config/systemd/user"
    d.mkdir(parents=True, exist_ok=True)
    (d / "goldbach-ci.service").write_text(f"""[Unit]
Description=goldbach-l0 local CI daemon (recovery, retries, count checks, branch watching)
After=default.target

[Service]
Type=simple
WorkingDirectory={ROOT}
ExecStart={sys.executable} {ROOT / 'ci' / 'goldbach_ci.py'} daemon
Restart=on-failure
RestartSec=60
# CI children die with the daemon; run_range runs in its own unit ({UNIT})
KillMode=control-group

[Install]
WantedBy=default.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    if a.enable:
        subprocess.run(["systemctl", "--user", "enable", "--now", "goldbach-ci.service"], check=True)
    print(f"wrote {d / 'goldbach-ci.service'}" + (" (enabled + started)" if a.enable else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    p = cmd("bootstrap", cmd_bootstrap, "first known-good release from HEAD; adopt the running run_range")
    p.add_argument("--rev")
    p.add_argument("--no-verify", action="store_true")
    p = cmd("status", cmd_status, "releases, production, what the CI is doing, last runs")
    p.add_argument("-n", type=int, default=10)
    cmd("poll", cmd_poll, "one pass: retry queue + new branch tips + count check")
    cmd("daemon", cmd_daemon, "recovery, then poll forever (the systemd service)")
    p = cmd("run", cmd_run, "CI for one revision now (GitHub job gpu)")
    p.add_argument("rev", nargs="?", default="main")
    p.add_argument("--force-gpu", action="store_true", help="GPU stages even for an identical artifact")
    p.add_argument("--rerun", action="store_true", help="test again even if REV already has a final result")
    p.add_argument("--skip-cpu-tests", action="store_true", help="kernel freshness + ctest already ran (Actions)")
    p.add_argument("--ref", help="branch name to record")
    p.add_argument("--fetch", action="store_true", help="git fetch origin REV first")
    p.add_argument("--summary", help="append report.md here ($GITHUB_STEP_SUMMARY)")
    p.add_argument("--artifacts", help="copy the run directory here")
    p = cmd("production", cmd_production, "start / stop [--now] / status / adopt run_range")
    p.add_argument("action", choices=["start", "stop", "status", "adopt"])
    p.add_argument("--now", action="store_true", help="stop: kill the slice in progress (it re-runs)")
    p = cmd("rollback", cmd_rollback, "previous release -> active, restart production")
    p.add_argument("--now", action="store_true")
    p = cmd("count-check", cmd_count_check, "banked slices' prime counts vs primecount")
    p.add_argument("--journal")
    p.add_argument("--all", action="store_true", help="re-check slices already checked")
    p.add_argument("--max", type=int, default=None)
    p = cmd("recheck", cmd_recheck, "re-run queued (or one) banked slice(s) and compare")
    p.add_argument("start", nargs="?", type=int)
    p.add_argument("--release")
    p = cmd("benchmark", cmd_benchmark, "benchmark session + figures + analysis (pauses production)")
    p.add_argument("--profile", choices=sorted(BENCH_PROFILES), default="full")
    p.add_argument("--summary")
    p.add_argument("--artifacts")
    p = cmd("install-service", cmd_install_service, "write (and --enable) the systemd user service")
    p.add_argument("--enable", action="store_true")
    a = ap.parse_args()
    # SIGTERM (systemctl stop, a cancelled Actions job) unwinds like Ctrl-C:
    # child process groups are killed, production is resumed
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    a.fn(a)


if __name__ == "__main__":
    main()
