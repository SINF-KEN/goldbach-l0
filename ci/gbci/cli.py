"""The command line: one cmd_* per subcommand, main()."""
import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .bench import benchmark_session
from .build import build_release
from .core import BENCH_PROFILES, FINAL, ROOT, STATE, UNIT, cfg, flock, git, load_json, log, log_exc, now_iso, save_json, stamp
from .daemon import poll_once, recover
from .gpustack import GPU_STACK_FAILED, gpu_stack
from .ledger import candidates, ledger, ledger_append, list_refs, mark_seen, queue_edit
from .audit import count_check, recheck
from .prodstate import find_run_range, prod_conf, prod_paths, save_prod_conf
from .production import adopt_production, last_run_range_event, production_status, start_production, stop_production
from .releases import active_release, fingerprint, link_target, promote, release_meta, set_link
from .report import annotate, heartbeat, progress_line
from .run import Run


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
            with contextlib.ExitStack() as held:
                try:
                    held.enter_context(flock(STATE / "ci.lock"))
                except RuntimeError:      # a CI run (Actions job, command) is busy: next poll
                    pass
                else:
                    try:
                        poll_once()
                    except Exception as e:  # the daemon outlives any one poll; the next one retries
                        log_exc(f"daemon: poll {type(e).__name__}: {e}", key="daemon-poll")
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


def sigterm_as_interrupt(signum, frame):
    """SIGTERM (systemctl stop, a cancelled Actions job) unwinds like Ctrl-C:
    child process groups are killed, production is resumed."""
    raise KeyboardInterrupt


def main(description=None):
    """Parse the command line and run the subcommand; `description` heads
    --help (goldbach_ci.py passes its own docstring)."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    signal.signal(signal.SIGTERM, sigterm_as_interrupt)
    a.fn(a)
