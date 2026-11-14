#!/usr/bin/env python3
"""ci/production_env.py: knob validation and the env plan (set, change,
unset of a key the file no longer lists, foreign keys left alone), and the
apply flow against a stub of the production helpers (stop -> save -> start;
restart refused -> previous env restored and production restarted)."""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ci"))
import production_env as pe  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print(("  ok   " if cond else "  FAIL ") + what)
    fails += 0 if cond else 1


def spec_ok(env):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"env": env}, f)
    try:
        pe.load_spec(f.name)
        return True
    except ValueError:
        return False


check(spec_ok({}), "empty file")
check(spec_ok({"GB_GAPS": "sampled", "GB_BUCKET_VERIFY": "500"}), "both knobs")
check(not spec_ok({"GB_GAPS": "fast"}), "GB_GAPS: unknown value refused")
check(not spec_ok({"GB_BUCKET_VERIFY": "0"}), "GB_BUCKET_VERIFY=0 refused")
check(not spec_ok({"GB_BUCKET_VERIFY": 500}), "a non-string value refused")
check(not spec_ok({"GB_RANGE_BIN": "/x"}), "GB_RANGE_BIN reserved")
check(not spec_ok({"GB_NTFY_TOPIC": "x"}), "notification secret reserved")
check(not spec_ok({"PATH": "/x"}), "non-GB_ key refused")
check(not spec_ok({"GB_X": "a b"}), "shell metacharacters refused")

cur = {"GB_BUCKET_VERIFY": "100", "GB_STATS_LOG": "s.log"}
new, ch = pe.plan(cur, [], {"GB_GAPS": "sampled"})
check(new == {"GB_BUCKET_VERIFY": "100", "GB_STATS_LOG": "s.log", "GB_GAPS": "sampled"} and
      ch == [("GB_GAPS", "exact", "sampled")], "set a new key, others kept")
new, ch = pe.plan(cur, [], {"GB_BUCKET_VERIFY": "500"})
check(new["GB_BUCKET_VERIFY"] == "500" and ch == [("GB_BUCKET_VERIFY", "100", "500")], "change a key")
new, ch = pe.plan({"GB_GAPS": "sampled", "GB_STATS_LOG": "s.log"}, ["GB_GAPS"], {})
check(new == {"GB_STATS_LOG": "s.log"} and ch == [("GB_GAPS", "sampled", "exact")], "unset a key no longer listed")
new, ch = pe.plan(cur, ["GB_X"], {})
check(new == cur and ch == [], "unlisted managed key already absent: no change")
new, ch = pe.plan({"GB_X": "1"}, ["GB_X"], {})
check(new == {} and ch == [("GB_X", "1", None)], "unset a key without a default")
new, ch = pe.plan({}, [], {"GB_BUCKET_VERIFY": "100", "GB_GAPS": "exact"})
check(new == {"GB_BUCKET_VERIFY": "100", "GB_GAPS": "exact"} and ch == [], "writing down the defaults: no change")
new, ch = pe.plan({"GB_BUCKET_VERIFY": "100"}, ["GB_BUCKET_VERIFY"], {})
check(new == {} and ch == [], "dropping a default-valued key: no change")


class Stub:
    """goldbach_ci stand-in: production.json in memory, start may refuse."""
    def __init__(self, env, running=True, refuse=None):
        self.pc = {"env": dict(env), "desired": "running"}
        self.running, self.refuse, self.starts = running, refuse, []

    def prod_conf(self):
        return json.loads(json.dumps(self.pc))

    def save_prod_conf(self, pc):
        self.pc = json.loads(json.dumps(pc))

    def find_run_range(self):
        return [1] if self.running else []

    def stop_production(self, reason=""):
        self.running = False
        return {"mode": "boundary", "next_after": 7}

    def start_production(self):
        env = dict(self.pc["env"])
        self.starts.append(env)
        if self.refuse and self.refuse(env):
            raise RuntimeError("run_range refused to start")
        self.running = True
        return {"release": "r1", "resumed_at": 7}

    def log(self, msg):
        pass

    def alert(self, title, msg):
        self.alerted = title


g = Stub({"GB_BUCKET_VERIFY": "100"})
rc = pe.apply(g, {"GB_BUCKET_VERIFY": "500"}, False, None)
check(rc == 0 and g.running and g.pc["env"]["GB_BUCKET_VERIFY"] == "500" and g.pc["env_managed"] == ["GB_BUCKET_VERIFY"]
      and len(g.starts) == 1, "apply: stop, save, one restart with the new env")
g = Stub({"GB_BUCKET_VERIFY": "100"}, refuse=lambda e: e.get("GB_GAPS") == "sampled")
rc = pe.apply(g, {"GB_GAPS": "sampled"}, False, None)
check(rc == 1 and g.running and "GB_GAPS" not in g.pc["env"] and g.pc.get("env_managed", []) == []
      and len(g.starts) == 2, "refused restart: previous env restored, production restarted")
g = Stub({}, running=False)
rc = pe.apply(g, {"GB_GAPS": "sampled"}, False, None)
check(rc == 0 and not g.running and g.pc["env"] == {"GB_GAPS": "sampled"} and not g.starts,
      "not running: saved, not started")
g = Stub({})
rc = pe.apply(g, {"GB_BUCKET_VERIFY": "100", "GB_GAPS": "exact"}, False, None)
check(rc == 0 and g.running and not g.starts and g.pc["env_managed"] == ["GB_BUCKET_VERIFY", "GB_GAPS"],
      "defaults written down: env saved, no restart")
g = Stub({"GB_BUCKET_VERIFY": "100"})
rc = pe.apply(g, {"GB_BUCKET_VERIFY": "500"}, True, None)
check(rc == 0 and g.pc["env"]["GB_BUCKET_VERIFY"] == "100" and not g.starts, "dry run touches nothing")

print(f"production_env: {fails} failure(s) -> {'PASS' if not fails else 'FAIL'}")
sys.exit(1 if fails else 0)
