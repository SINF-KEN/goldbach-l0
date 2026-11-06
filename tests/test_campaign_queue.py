#!/usr/bin/env python3
"""ci/campaign_queue.py: a finished range hands over to the next queued one;
a running, stopped or failed range, or an empty queue, is left alone."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q = HERE.parent / "ci" / "campaign_queue.py"
fails = 0


def check(cond, what):
    global fails
    print(("  ok   " if cond else "  FAIL ") + what)
    fails += 0 if cond else 1


def setup(last_line, then):
    root = Path(tempfile.mkdtemp())
    (root / "ci-state").mkdir()
    (root / "ci").mkdir()
    # stub CI: records that production start was asked for
    (root / "ci" / "goldbach_ci.py").write_text("import sys, pathlib\npathlib.Path(sys.argv[0]).with_name('started').write_text(' '.join(sys.argv[1:]))\n")
    pc = {"start": "4000000000000000000", "limit": "4113000000000000000", "slice": "50000000000000",
          "journal": "range-4000000000000000000-4113000000000000000.journal", "desired": "running", "env": {}, "then": then}
    (root / "ci-state" / "production.json").write_text(json.dumps(pc))
    (root / "range-4000000000000000000-4113000000000000000.log").write_text(
        "2026-10-19T03:00:00+02:00 run_range: [4000000000000000000, 4113000000000000000] slice 50000000000000, resuming at 4112950000000000000 release=x\n" + last_line + "\n")
    return root


def run(root, *extra):
    # the real production run on this machine must not count as "running" here
    env = dict(os.environ, GB_QUEUE_PROC_RE="^no-such-process-campaign-queue-test$")
    return subprocess.run([sys.executable, str(Q), "--root", str(root), *extra], capture_output=True, text=True, env=env)


MAIN = {"start": "4000000000000000000", "limit": "5000000000000000000", "slice": "50000000000000",
        "journal": "range-4000000000000000000-5000000000000000000.journal"}
done = "2026-10-27T03:00:00+02:00 COMPLETE: every even in [4000000000000000000, 4113000000000000000] verified (journal x)"
r = setup(done, [MAIN])
run(r)
pc = json.loads((r / "ci-state" / "production.json").read_text())
check(pc["limit"] == MAIN["limit"] and pc["journal"] == MAIN["journal"] and pc["then"] == [] and
      (r / "ci" / "started").read_text() == "production start", "a complete range hands over to the queued range")
check(oct(os.stat(r / "ci-state" / "production.json").st_mode & 0o777) == "0o600", "production.json stays private (0600)")
r = setup("2026-10-27T03:00:00+02:00 STOPPED: flaky hardware", [MAIN])
run(r)
check(not (r / "ci" / "started").exists() and json.loads((r / "ci-state" / "production.json").read_text())["then"] == [MAIN],
      "a stopped range is left to the CI")
r = setup(done, [])
run(r)
check(not (r / "ci" / "started").exists(), "an empty queue does nothing")
r = setup(done, [MAIN])
run(r, "--dry-run")
check(not (r / "ci" / "started").exists(), "--dry-run changes nothing")
r = setup(done, [])
run(r, "--push", "1", "3", "--slice", "2")
check(json.loads((r / "ci-state" / "production.json").read_text())["then"][0]["journal"] == "range-1-3.journal", "--push queues a range first")
print("FAIL" if fails else "campaign queue OK")
sys.exit(1 if fails else 0)
