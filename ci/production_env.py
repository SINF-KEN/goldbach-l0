#!/usr/bin/env python3
"""production_env.py — production's GB_* knobs as code (stdlib only).

ci/production-env.json lists the GB_* settings the repository manages in
production's environment (ci-state/production.json "env", the environment
run_range runs with):

  {"env": {"GB_GAPS": "sampled", "GB_BUCKET_VERIFY": "500"}}

Run by the GitHub workflow `Production env` (.github/workflows/
production-env.yml) when a push to main changes that file, or by hand:

  python3 ci/production_env.py ci/production-env.json --prod-dir DIR [--dry-run]

Every key in the file is set to its value; a key the file managed before
(production.json "env_managed") and no longer lists is removed, so run_range's
own default applies again — reverting the commit reverts the knob. Keys the
file never listed are left alone. With a change and production running it is
stopped at the next slice boundary (the production tree's goldbach_ci.py:
ci.lock, journal audit), the environment is saved and production restarted
(resume point + release verified). If the restart fails the previous
environment is restored and production restarted with it; the job fails.
"""
import argparse
import json
import re
import signal
import sys
from pathlib import Path

KEY_RE = re.compile(r"^GB_[A-Z0-9_]+$")
VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:+-]{1,64}$")
# set by start_production / run_range per start, or secrets: never managed here
RESERVED = {"GB_RANGE_BIN", "GB_RANGE_SPV", "GB_RANGE_TIMEOUT", "GB_NTFY_TOPIC", "GB_NOTIFY_MAIL",
            "GB_NOTIFY_CMD", "GB_NOTIFY_WALL", "GB_RECORD_HOOK"}
# the value an absent key means (run_range.sh exports GB_BUCKET_VERIFY=100;
# goldbach_l0 walks every gap without GB_GAPS)
DEFAULTS = {"GB_BUCKET_VERIFY": "100", "GB_GAPS": "exact"}
KNOWN = {
    "GB_GAPS": lambda v: v in ("exact", "sampled", "0"),
    "GB_BUCKET_VERIFY": lambda v: v.isdigit() and 1 <= int(v) <= 1_000_000,
}


def load_spec(path):
    """-> {key: value}; raises ValueError on anything not a valid GB_* knob."""
    spec = json.loads(Path(path).read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("env"), dict):
        raise ValueError(f'{path}: expected {{"env": {{...}}}}')
    env = spec["env"]
    for k, v in env.items():
        if not KEY_RE.match(k):
            raise ValueError(f"{k}: not a GB_* name")
        if k in RESERVED or k.startswith("GB_NOTIFY"):
            raise ValueError(f"{k}: reserved (set per start, or a secret)")
        if not isinstance(v, str) or not VALUE_RE.match(v):
            raise ValueError(f"{k}={v!r}: the value must be a short string of [A-Za-z0-9_.,:+-]")
        if k in KNOWN and not KNOWN[k](v):
            raise ValueError(f"{k}={v}: not an accepted value")
    return dict(env)


def plan(cur_env, managed, desired):
    """-> (new env, [(key, old, new)]) for production.json. A change is a
    change of the EFFECTIVE value (absent = run_range / goldbach_l0's
    default, DEFAULTS): writing down a default, or dropping a key that held
    it, edits the env without a restart."""
    eff = lambda env, k: env.get(k, DEFAULTS.get(k))  # noqa: E731
    new = dict(cur_env)
    for k in set(managed) - set(desired):
        new.pop(k, None)
    for k, v in desired.items():
        new[k] = v
    changes = [(k, eff(cur_env, k), eff(new, k)) for k in sorted(set(cur_env) | set(new))
               if eff(cur_env, k) != eff(new, k)]
    return new, changes


def fmt(changes):
    return ", ".join(f"{k}: {'unset' if o is None else o} -> {'unset' if n is None else n}"
                     for k, o, n in changes) or "no change"


def apply(g, desired, dry_run, summary):
    def report(msg):
        print(msg, flush=True)
        if summary:
            with open(summary, "a") as f:
                f.write(msg + "\n\n")

    pc = g.prod_conf()
    if not pc:
        raise SystemExit("no ci-state/production.json (production never adopted)")
    cur_env, cur_managed = pc.get("env", {}), pc.get("env_managed", [])
    new_env, changes = plan(cur_env, cur_managed, desired)
    report(f"**Production env** (managed: {', '.join(sorted(desired)) or 'none'}): {fmt(changes)}")
    if dry_run:
        return 0
    if not changes:
        if new_env != cur_env or sorted(cur_managed) != sorted(desired):
            pc["env"], pc["env_managed"] = new_env, sorted(desired)
            g.save_prod_conf(pc)            # same effective values: no restart
        return 0
    running = bool(g.find_run_range())
    stopped = False
    try:
        if running:
            info = g.stop_production(reason="production env: " + fmt(changes))
            stopped = True
            report(f"production stopped ({info.get('mode')}), resume point {info.get('next_after')}")
        pc = g.prod_conf()                      # stop_production may have re-saved it
        old_env, old_managed = dict(pc.get("env", {})), list(pc.get("env_managed", []))
        pc["env"], changes = plan(old_env, old_managed, desired)
        pc["env_managed"] = sorted(desired)
        g.save_prod_conf(pc)
        g.log(f"production env: {fmt(changes)}")
        if not stopped:
            report("production was not running: saved, it applies at the next start")
            return 0
        try:
            res = g.start_production()
            stopped = False
            report(f"production restarted with the new env: release {res['release']}, "
                   f"resumed at {res['resumed_at']} (verified)")
            return 0
        except Exception as e:  # noqa: BLE001 — any refusal: put the old env back
            g.log(f"production env: restart refused ({e}); restoring the previous env")
            pc = g.prod_conf()
            pc["env"], pc["env_managed"] = old_env, old_managed
            g.save_prod_conf(pc)
            g.alert("production env change refused", f"{fmt(changes)}: {e}; previous env restored")
            report(f"**restart with the new env FAILED** ({e}); previous env restored")
            return 1
    finally:
        if stopped:                             # never leave production down
            try:
                res = g.start_production()
                report(f"production restarted: release {res['release']}, resumed at {res['resumed_at']}")
            except Exception as e:  # noqa: BLE001
                g.log(f"production env: could NOT restart production: {e}")
                g.alert("production env: could not restart run_range", str(e))
                report(f"**could NOT restart production**: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--prod-dir", type=Path, help="production goldbach-l0 tree (omit with --check)")
    ap.add_argument("--check", action="store_true", help="validate the file only")
    ap.add_argument("--dry-run", action="store_true", help="print the change, touch nothing")
    ap.add_argument("--summary", type=Path)
    a = ap.parse_args()
    try:
        desired = load_spec(a.spec)
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: {e}")
    if a.check:
        print(f"{a.spec}: ok ({', '.join(f'{k}={v}' for k, v in sorted(desired.items())) or 'empty'})")
        return 0
    if not a.prod_dir:
        ap.error("--prod-dir is required")
    sys.path.insert(0, str(a.prod_dir / "ci"))
    import goldbach_ci as g                     # the production tree's helpers (ROOT, STATE)
    # SIGTERM (a cancelled job) unwinds like Ctrl-C: the finally restarts production
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with g.heartbeat(60), g.flock(g.STATE / "ci.lock", blocking=True, timeout=6 * 3600):
        return apply(g, desired, a.dry_run, a.summary)


if __name__ == "__main__":
    sys.exit(main())
