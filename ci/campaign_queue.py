#!/usr/bin/env python3
"""campaign_queue.py — run production ranges one after another, unattended.

ci-state/production.json may hold `"then": [{"start", "limit", "slice",
"journal"}, ...]`: ranges to run after the current one. When run_range has
finished the current range (its log's last run_range line is COMPLETE, it is
not running, production is desired running), the first queued range becomes
the current one and `goldbach_ci.py production start` starts it. Anything else
(running, stopped, failed, queue empty) is left alone. Run by
goldbach-campaign-queue.timer every 5 minutes; safe to run at any time.

  python3 ci/campaign_queue.py [--root DIR] [--dry-run]
  python3 ci/campaign_queue.py --push START LIMIT [--slice N]   # queue a range at the front
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def load(p):
    return json.loads(p.read_text())


def save(p, d):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def running():
    """A run_range.sh or goldbach_l0 process exists."""
    pat = os.environ.get("GB_QUEUE_PROC_RE", r"^(/usr/bin/)?bash \S*run_range\.sh |^\S*goldbach_l0 ")
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout
    return bool(out.strip())


SYNC_UNITS = ("goldbach-sync.timer", "goldbach-sync.path")


def ensure_sync(dry_run=False):
    """Production running without the site sync (started by hand after a boot,
    2026-10-23): start the sync too. Only ever started next to a running range,
    never enabled, so nothing runs at boot that the user did not start."""
    if not running():
        return
    down = [u for u in SYNC_UNITS
            if subprocess.run(["systemctl", "--user", "is-active", "--quiet", u]).returncode != 0]
    if down:
        print(f"production runs without {' '.join(down)}: starting {'(dry run)' if dry_run else ''}")
        if not dry_run:
            subprocess.run(["systemctl", "--user", "start", *down])


def last_event(log):
    try:
        lines = [ln for ln in log.read_text(errors="replace").splitlines() if "run_range" in ln or "COMPLETE" in ln]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/coding/goldbach-l0-tidy"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", nargs=2, metavar=("START", "LIMIT"))
    ap.add_argument("--slice", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    pj = root / "ci-state" / "production.json"
    pc = load(pj)
    if a.push:
        s, l = a.push
        item = {"start": s, "limit": l, "slice": a.slice or pc["slice"], "journal": f"range-{s}-{l}.journal"}
        pc["then"] = [item] + pc.get("then", [])
        save(pj, pc)
        print(f"queued first: {item}")
        return 0
    ensure_sync(a.dry_run)
    queue = pc.get("then", [])
    if not queue or pc.get("desired") != "running" or running():
        return 0
    log = root / pc["journal"].replace(".journal", ".log")
    ev = last_event(log)
    if not re.search(rf"COMPLETE: every even in \[{pc['start']}, {pc['limit']}\]", ev):
        return 0                                   # stopped or failed: the CI's business, not ours
    nxt = queue[0]
    print(f"range [{pc['start']}, {pc['limit']}] complete; next: [{nxt['start']}, {nxt['limit']}]")
    if a.dry_run:
        return 0
    pc.update({k: nxt[k] for k in ("start", "limit", "slice", "journal")})
    pc["then"] = queue[1:]
    save(pj, pc)
    p = subprocess.run([sys.executable, str(root / "ci" / "goldbach_ci.py"), "production", "start"],
                       capture_output=True, text=True)
    print(p.stdout.strip(), p.stderr.strip())
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
