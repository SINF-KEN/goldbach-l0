"""The daemon's work: crash / reboot recovery and one poll (count check, GPU
stack re-verification, retry queue, new branch tips, auto recheck)."""
import contextlib
import os
import re
import shutil
import signal

from .audit import count_check, recheck
from .core import STATE, cfg, load_json, log, log_exc, now_iso, save_json
from .gpustack import gpu_stack_check_due
from .ledger import attempts, candidates, judged, ledger_append, mark_seen, queue_edit
from .primecount import primecount_bin
from .prodstate import find_run_range, pgrep, proc_cmdline, prod_conf, prod_paths
from .production import last_run_range_event, start_production
from .releases import active_release, release_meta
from .report import alert
from .run import Run


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
            except Exception as e:      # production stays down: log and alert, the daemon goes on
                log_exc(f"recovery: resume failed: {e}")
                alert("CI recovery could not resume run_range", str(e))


def poll_once():
    c = cfg()
    if c["count_check"] and primecount_bin() and prod_conf():
        try:
            count_check(stop_on_mismatch=c["count_check_stop"], limit_blocks=2)
        except Exception as e:          # a broken count check must not hold up the CI runs below
            log_exc(f"count-check error: {e}", key="count-check")
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
