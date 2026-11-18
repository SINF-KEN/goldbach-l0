#!/usr/bin/env python3
"""gstats_fix.py — replace one slice's prime / twin counts in the campaign
statistics state with those of a verified re-run.

A transient GPU fault can leave a slice's map short of a few primes while its
verification still stands (a false composite only hides witnesses): the slice
PASSes, the CI count check flags it, `goldbach_ci.py recheck` re-runs it and
appends a RECHECK row with the primecount-matching counts. The journal readers
(the site) take those counts, but the statistics state goldbach_stats.state
kept the faulty slice's primes30 / twins30 (2026-10-03: 21 primes and 2 twins
short). This tool applies the difference:

  scripts/gstats_fix.py --state goldbach_stats.state --log goldbach_stats.log \\
      --rerun RERUN_LOG [--primecount N] [--apply]

RERUN_LOG is the isolated GB_STATS_LOG of a re-run of exactly the slice (same
start / limit and, for an identical sampled histogram, the original
--segment). The tool refuses unless the re-run's GBSTATS line agrees with the
original's in every field the state cannot correct from the log (evens,
sampled evens, tail threshold, tail counts, tail evens, largest p(n), CPU
records beyond 9781, and every first occurrence the original introduced):
only primes / twins and their residue-class splits may differ. With
--primecount the re-run's prime count must equal it. Without --apply it
prints the correction only.

--apply (under the state's lock, like a verifier run): a timestamped backup of
the state, the corrected state written atomically (temp file, fsync, rename),
and a "GBFIX v1" line appended to the log recording the slice, the per-class
deltas and the re-run, so the state stays reproducible from the log.
Exit status: 0 applied / nothing to do / dry run, 1 refused, 2 usage.
"""
import argparse
import fcntl
import os
import shutil
import sys
import time

FIXED_FIELDS = ("evens", "evens_sampled", "tail_t", "tail_evens", "pmax", "tail", "beyond")


def parse_line(line):
    return dict(x.split("=", 1) for x in line.split()[2:] if "=" in x)


def gbstats_lines(path, start, limit):
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith("GBSTATS v2 "):
                kv = parse_line(line)
                if (start is None or kv.get("start") == start) and (limit is None or kv.get("limit") == limit):
                    out.append(kv)
    return out


def firsts(kv):
    v = kv.get("new_first", "-")
    return {} if v in ("", "-") else dict(e.split(":", 1) for e in v.split(";"))


def correction(state_path, log_path, rerun_path, primecount=None):
    """@return (delta primes30, delta twins30, original kv, re-run kv, problems)."""
    new = gbstats_lines(rerun_path, None, None)
    if len(new) != 1:
        return None, None, None, None, [f"{rerun_path}: expected one GBSTATS v2 line, found {len(new)}"]
    new = new[0]
    old = gbstats_lines(log_path, new["start"], new["limit"])
    if len(old) != 1:
        return None, None, None, new, [f"{log_path}: expected one GBSTATS line for [{new['start']}, {new['limit']}], "
                                       f"found {len(old)}"]
    old = old[0]
    bad = [f"{k}: original {old.get(k)!r} != re-run {new.get(k)!r}" for k in FIXED_FIELDS if old.get(k) != new.get(k)]
    nf = firsts(new)
    bad += [f"first occurrence of p = {p}: original {n}, re-run {nf.get(p)}" for p, n in firsts(old).items()
            if nf.get(p) != n]
    if primecount is not None and int(new["primes"]) != primecount:
        bad.append(f"re-run primes {new['primes']} != primecount {primecount}")
    p_old = [int(x) for x in old["primes30"].split(",")]
    p_new = [int(x) for x in new["primes30"].split(",")]
    t_old = [int(x) for x in old["twins30"].split(",")]
    t_new = [int(x) for x in new["twins30"].split(",")]
    if sum(p_new) != int(new["primes"]) or sum(t_new) != int(new["twins"]):
        bad.append("re-run line inconsistent (class sums != totals)")
    return ([b - a for a, b in zip(p_old, p_new)], [b - a for a, b in zip(t_old, t_new)], old, new, bad)


def rewrite_state(path, dp, dt):
    lines = open(path).read().splitlines(keepends=True)
    seen = set()
    for i, ln in enumerate(lines):
        w = ln.split()
        if w and w[0] in ("primes30", "twins30"):
            d = dp if w[0] == "primes30" else dt
            vals = [int(x) for x in w[1:]]
            if len(vals) != len(d):
                raise ValueError(f"state line {w[0]} has {len(vals)} fields, expected {len(d)}")
            vals = [v + x for v, x in zip(vals, d)]
            if min(vals) < 0:
                raise ValueError(f"state line {w[0]} would go negative")
            lines[i] = w[0] + " " + " ".join(map(str, vals)) + "\n"
            seen.add(w[0])
    if seen != {"primes30", "twins30"}:
        raise ValueError(f"state {path} lacks {sorted({'primes30', 'twins30'} - seen)}")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--rerun", required=True, help="isolated GB_STATS_LOG of the re-run")
    ap.add_argument("--primecount", type=int, help="primecount's count for the slice (must match the re-run)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--reason", default="recheck")
    a = ap.parse_args()

    dp, dt, old, new, bad = correction(a.state, a.log, a.rerun, a.primecount)
    if bad:
        print("refused:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print(f"slice [{new['start']}, {new['limit']}]: original {old['version']} primes {old['primes']} twins "
          f"{old['twins']}; re-run {new['version']} primes {new['primes']} twins {new['twins']}")
    print(f"  delta primes30 {dp} (total {sum(dp):+d}), twins30 {dt} (total {sum(dt):+d})")
    if not any(dp) and not any(dt):
        print("  nothing to correct")
        return 0
    with open(a.log) as f:
        if any(ln.startswith("GBFIX v1 ") and f"start={new['start']} limit={new['limit']} " in ln for ln in f):
            print("  already applied (GBFIX line in the log)")
            return 0
    if not a.apply:
        print("  dry run: --apply to write it")
        return 0
    with open(a.state + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        backup = f"{a.state}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(a.state, backup)
        rewrite_state(a.state, dp, dt)
        reason = "".join(c if c.isalnum() or c in "-_.+" else "_" for c in a.reason)   # one key=value token
        with open(a.log, "a") as f:
            f.write(f"GBFIX v1 start={new['start']} limit={new['limit']} reason={reason} "
                    f"original_version={old['version']} rerun_version={new['version']} "
                    f"primes={old['primes']}->{new['primes']} twins={old['twins']}->{new['twins']} "
                    f"dprimes30={','.join(map(str, dp))} dtwins30={','.join(map(str, dt))} "
                    f"at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
            f.flush()
            os.fsync(f.fileno())
    print(f"  applied (backup {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
