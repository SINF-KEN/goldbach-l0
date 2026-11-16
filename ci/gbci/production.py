"""Production control: adopt, status, stop (slice boundary or now), start
(verified resume point and release) and production_paused, which resumes
production on every exit path."""
import contextlib
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from .core import ROOT, RUN_RANGE, STATE, UNIT, cfg, log, log_exc, save_json, set_phase
from .journal import chain_next, journal_audit, median_slice_wall, read_journal
from .prodstate import find_run_range, pgrep, prod_conf, prod_live, prod_paths, save_prod_conf
from .releases import active_release, release_meta
from .report import alert


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
            except Exception as e:      # whatever failed: record, log, alert; never mask the body's outcome
                info["resume_error"] = str(e)
                log_exc(f"could NOT resume production: {e}")
                alert("CI could not resume run_range", str(e))
