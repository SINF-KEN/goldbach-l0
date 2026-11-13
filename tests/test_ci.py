#!/usr/bin/env python3
"""test_ci.py — GPU-free tests of the local CI's logic (ci/goldbach_ci.py):
journal chain rules, reference-cache keys, result comparison, benchmark
analysis, ledger layout, retry queue. Uses a throwaway GB_CI_STATE.

    python3 tests/test_ci.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="gbci-test-")
os.environ["GB_CI_STATE"] = TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ci"))
import goldbach_ci as g  # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    fails += not ok


def journal(lines):
    p = Path(TMP) / "t.journal"
    p.write_text("\n".join(lines) + "\n")
    return g.read_journal(p)


# --- journal: the resume chain, retries, duplicates -----------------------
rows = journal(["PASS 100 102 2026-10-05T10:00:00+02:00 limit=102 start=100 primes=3 release=a",
                "RETRY 104 106 2026-10-05T10:10:00+02:00 attempt=1 rc=1 reason=x y z",
                "PASS 104 106 2026-10-05T10:20:00+02:00 limit=106 start=104 primes=2 attempts=2",
                "RECHECK 100 102 2026-10-05T11:00:00+02:00 outcome=CONFIRMED",
                "PASS 110 112 2026-10-05T10:30:00+02:00 limit=112 start=110 primes=1"])
check(g.chain_next(rows, 100) == 108, "chain stops at the gap 108 (110 is not contiguous)")
check(g.chain_next(rows, 104) == 108, "chain from another START")
check([r["tag"] for r in rows] == ["PASS", "RETRY", "PASS", "PASS"], "RECHECK rows are not chain rows")
nxt, dups = g.journal_audit(journal(["PASS 100 102 t limit=102 start=100", "PASS 100 102 t limit=102 start=100",
                                     "PASS 104 106 t limit=106 start=104"]), 100)
check(nxt == 108 and dups == [(100, 102)], "journal_audit finds a duplicate PASS row")
# a PASS row whose slice end b disagrees with its own --csv limit= would skip
# unverified evens: the chain must refuse it, not follow it
damaged = journal(["PASS 100 102 t limit=102 start=100", "PASS 104 900 t limit=106 start=104"])
try:
    g.chain_next(damaged, 100)
    check(False, "chain refuses a PASS row whose b disagrees with its limit=")
except RuntimeError:
    check(True, "chain refuses a PASS row whose b disagrees with its limit=")
# a row cut off by a crash (non-numeric field) is ignored, not a crash
cut = journal(["PASS 100 102 t limit=102 start=100", "PASS 104 10PASS 108 110 t limit=110 start=108"])
check(g.chain_next(cut, 100) == 104 and len(cut) == 1, "a malformed (cut-off) row is ignored")
check(g.kv("a=1 b=x=y junk c=") == {"a": "1", "b": "x=y", "c": ""}, "kv parses key=value tokens")

# --- primality used for pi(odd x) --------------------------------------------
small = [n for n in range(2, 5000) if all(n % d for d in range(2, int(n ** 0.5) + 1))]
check([n for n in range(5000) if g.is_prime(n)] == small, "is_prime == trial division below 5000")
check(not g.is_prime(3825123056546413051) and g.is_prime(18446744073709551557),
      "is_prime: strong pseudoprime to bases 2..23 rejected, 2^64 - 59 accepted")

# --- reference cache key ------------------------------------------------------
env = {"GB_NTFY_TOPIC": "secret", "GB_RANGE_TIMEOUT": "5000", "GB_NOTIFY_SP": "1"}
k0 = g.ref_key("fp", 4, 10, "425000000", env)
check(k0 == g.ref_key("fp", 4, 10, "425000000", {}), "notification / GB_RANGE_* settings do not change the key")
check(k0 == g.ref_key("fp", 4, 10, "425000000", {"GB_BUCKET_VERIFY": "100"}), "GB_BUCKET_VERIFY=100 is the default")
check(len({k0, g.ref_key("fp2", 4, 10, "425000000", {}), g.ref_key("fp", 4, 12, "425000000", {}),
           g.ref_key("fp", 4, 10, "350000000", {}), g.ref_key("fp", 4, 10, "425000000", {"GB_STATS": "0"})}) == 5,
      "code, window, segment and result-relevant GB_* settings all change the key")

# --- compare(): the bit-exact gate ------------------------------------------
base = {"result": "PASS", "leftovers": "0", "segments": "47", "primes": "1000", "twins": "50",
        "stats_tail": "7", "stats_sampled": "9", "stats_pmax": "3001", "start": "4", "limit": "10", "gpu_s": "1.0"}
line = "GBSTATS v2 version=2.2.1 start=4 limit=10 primes=1000 tail=3001:1 new_first=-"
res = {"rc": 0, "csv": dict(base), "stats_line": line}
ref = {"kv": dict(base, gpu_s="0.9"), "stats_line": line.replace("2.2.1", "2.2.0").replace("new_first=-", "new_first=3001:5")}
lv = {x["name"]: x["level"] for x in g.compare(res, ref, 1000)}
check(all(v == "ok" for v in lv.values()) and len(lv) == 13,
      "identical results pass every exact check (version / new_first / timings ignored)")
bad = dict(res, csv=dict(base, primes="999"))
lv = {x["name"]: x["level"] for x in g.compare(bad, ref, 1000)}
check(lv["exact_primes"] == "fail" and lv["primecount"] == "fail", "a lost prime fails exact_primes and primecount")
bad = dict(res, stats_line=line.replace("tail=3001:1", "tail=3001:2"))
check({x["name"]: x["level"] for x in g.compare(bad, ref, 1000)}["exact_stats_line"] == "fail",
      "a different p(n) histogram fails exact_stats_line")
check({x["name"]: x["level"] for x in g.compare(res, ref, None)}["primecount"] == "warn",
      "missing primecount is a warning, not a pass")
check({x["name"]: x["level"] for x in g.compare(dict(res, rc=1), ref, 1000)}["slice_exit"] == "fail",
      "a non-zero exit fails")

# --- benchmark analysis attributes the time to phases ------------------------
mk = lambda label, gpu, check_s: {"label": label, "window": "long", "status": "pass", "gpu_s": gpu, "init_s": "0.1",
                                  "sieve_s": "0.2", "large_s": "0.5", "check_s": check_s, "sync_s": "0.01",
                                  "evens_per_sec": str(5e10 / float(gpu))}
txt = "\n".join(g.bench_analysis([mk("new", "0.95", "0.15"), mk("old", "0.90", "0.10")], "new", ["long"]))
check("SLOWER than old by +5.56%" in txt and "check +50.0 ms" in txt, "analysis: slower, and why (check +50 ms)")
txt = "\n".join(g.bench_analysis([mk("new", "0.900", "0.10"), mk("old", "0.901", "0.10")], "new", ["long"]))
check("no meaningful difference" in txt, "analysis: < 0.2% is no meaningful difference")

# --- ledger: fixed results.csv layout ------------------------------------------
g.ledger_append({"run_id": "r1", "status": "PASS", "commit": "abc", "checks": [],
                 "bench": {"windows": {"long": {"speed_ratio": 1.001}}}})
g.ledger_append({"run_id": "r2", "status": "FAIL", "commit": "def", "reason": "x",
                 "checks": [{"name": "exact_primes", "level": "fail", "detail": ""}]})
import csv  # noqa: E402
hdr, *data = list(csv.reader(open(Path(TMP) / "results.csv")))
check(hdr == g.RESULT_COLS and all(len(r) == len(hdr) for r in data), "results.csv: one fixed header, aligned rows")
check(data[1][hdr.index("checks_failed")] == "exact_primes" and data[0][hdr.index("bench_long_speed_ratio")] == "1.001",
      "results.csv: values land in their columns")
check(g.judged("abc") and g.judged("def") and not g.judged("zzz"), "judged(): final statuses only")
g.ledger_append({"run_id": "r3", "status": "INTERRUPTED", "commit": "zzz", "checks": []})
check(g.attempts("zzz") == 1 and not g.judged("zzz"), "an interrupted run is an attempt, not a verdict")

# --- retry queue ----------------------------------------------------------------
g.queue_edit(lambda q: q + [{"sha": "zzz", "refs": ["main"], "reason": "retry", "added": "t"}])
g.queue_edit(lambda q: [x for x in q if x["sha"] != "zzz"])
check(json.loads((Path(TMP) / "queue.json").read_text()) == [], "queue add / remove")
check(g.progress_line() == "idle", "progress_line with no CI run")

# --- pausing production: stop now or wait for the boundary -------------------
check(g.stop_now_is_cheaper(60, 0.03, 300), "28-min slice, 1 min in: stop now (redo 1 min, not wait 27)")
check(not g.stop_now_is_cheaper(400, 0.25, 300), "redo above stop_max_loss_s: wait")
check(not g.stop_now_is_cheaper(240, 0.6, 300), "7-min slice, 4 min in: wait 3 min rather than redo 4")
check(g.stop_now_is_cheaper(100, 0.25, 300), "7-min slice, 1.7 min in: stop now (redo 1.7, not wait 5)")

# --- count check: blocks of consecutive slices, localisation ---------------
S0, SL = 4000000000000000000, 50000000000000
truth = {S0 + i * SL: 1000 + i for i in range(10)}          # primes per slice
# one prime between two slices (slice 1 ends at b, the next starts at b + 2):
# in no slice, but primecount over the block sees it (be1f397)
between = S0 + 2 * SL - 1
truth[between] = 1
g.primes_in = lambda a, b: sum(v for k, v in truth.items() if a <= k <= b)
g.is_prime = lambda n: n == between                          # this synthetic world's slice boundaries
g.alert = lambda *a: None
jp = Path(TMP) / "cc.journal"
(Path(TMP) / "production.json").write_text(json.dumps({"start": str(S0), "journal": str(jp), "env": {}}))
lines = [f"PASS {S0 + i * SL} {S0 + (i + 1) * SL - 2} t primes={truth[S0 + i * SL] - (i == 5)}" for i in range(9)]
jp.write_text("\n".join(lines) + "\n")
(Path(TMP) / "count-check.jsonl").unlink(missing_ok=True)
n, bad = g.count_check(journal=jp)
check(n == 2, "9 slices of 5e13 -> 2 full blocks of 2e14 checked, the trailing slice waits")
check(json.loads((Path(TMP) / "count-check.jsonl").read_text().splitlines()[0])["between"] == 1,
      "a prime between two slices of a block is not a GPU miss")
check([b_["a"] for b_ in bad] == [S0 + 5 * SL], "a wrong slice inside a block is localised")
n, bad = g.count_check(journal=jp)
check(n == 1 and not bad, "a known mismatching slice is not re-flagged; its good neighbours are re-checked")
n, _ = g.count_check(journal=jp, partial=True)
check(n == 1, "partial=True also checks the trailing short block")
check(g.count_check(journal=jp, partial=True) == (0, []), "everything resolved: nothing to do")
jp.write_text(jp.read_text() + f"PASS {S0 + 9 * SL} {S0 + 10 * SL - 2} t primes={truth[S0 + 9 * SL]}\n"
              + f"RECHECK {S0 + 9 * SL} {S0 + 10 * SL - 2} t outcome=CONFIRMED\n")
check(g.count_check(journal=jp, partial=True) == (0, []), "a slice with a RECHECK row counts as resolved")
# a second journal over the same integers (a backfill) is its own GPU run: the
# first journal's blocks do not cover it; rows from before the "journal" field
# cover only slices of the releases they name
jb = Path(TMP) / "cc-backfill.journal"
jb.write_text("".join(f"PASS {S0 + i * SL} {S0 + (i + 1) * SL - 2} t primes={truth[S0 + i * SL]} release=new\n"
                      for i in range(4)))
with open(Path(TMP) / "count-check.jsonl", "a") as f:
    f.write(json.dumps({"a": S0, "b": S0 + 4 * SL - 2, "slices": 4, "ok": True, "releases": ["old"]}) + "\n")
n, bad = g.count_check(journal=jb)
check(n == 1 and not bad, "a backfill journal is checked although another journal covered its integers")
check(json.loads((Path(TMP) / "count-check.jsonl").read_text().splitlines()[-1])["journal"] == jb.name,
      "count-check rows name their journal")
check(g.count_check(journal=jb) == (0, []), "the backfill's own OK block resolves it")
jl = Path(TMP) / "cc-legacy.journal"
jl.write_text("".join(f"PASS {S0 + i * SL} {S0 + (i + 1) * SL - 2} t primes={truth[S0 + i * SL]} release=old\n"
                      for i in range(4)))
check(g.count_check(journal=jl) == (0, []), "a row without a journal still covers slices of its releases")

# --- the GBSTATS gap fields a v2.4 candidate adds are exactly the allowed ones --
import re as _re
gh = (Path(__file__).resolve().parent.parent / "include" / "gstats.hpp").read_text()
body = gh[gh.index("std::string gap_fields() const"):gh.index("/// Appends @p line to the run log.")]
written = set(_re.findall(r'" (g[a-z0-9]*)="', body))
check(written == g.STATS_NEW_OK, f"STATS_NEW_OK == the gap fields gstats.hpp writes ({sorted(written)})")

# --- GPU stack (driver / kernel / VRAM) changes ------------------------------
stack = {"kernel": "7.2.9", "intel-graphics-compiler": "2.41.5", "mem_speed": "20000"}
g.gpu_stack = lambda: dict(stack)
(Path(TMP) / "gpu-stack.json").unlink(missing_ok=True)
check(g.gpu_stack_change() == {} and (Path(TMP) / "gpu-stack.json").exists(), "first look: baseline, no change")
stack["intel-graphics-compiler"] = "2.42.0"
stack["mem_speed"] = "19000"
ch = g.gpu_stack_change()
check(ch == {"intel-graphics-compiler": ("2.41.5", "2.42.0"), "mem_speed": ("20000", "19000")},
      "an IGC update and a VRAM speed change are detected")
check(g.gpu_stack_change() == ch, "the change stays pending until a run verifies the new stack")
check(g.gpu_stack_check_due() == ch, "a pending change is due for verification")
g.save_json(Path(TMP) / g.GPU_STACK_FAILED, dict(stack))
check(g.gpu_stack_check_due() == {}, "a stack whose re-verification failed is not re-run on every poll")
stack["kernel"] = "7.2.10"
check("kernel" in g.gpu_stack_check_due(), "... but a further change is verified again")
(Path(TMP) / g.GPU_STACK_FAILED).unlink()

# --- production's statistics log follows its GB_STATS_LOG ---------------------
check(g.prod_stats_log({"env": {"GB_STATS_LOG": "x/s.log"}}) == g.ROOT / "x/s.log"
      and g.prod_stats_log({}) == g.ROOT / "goldbach_stats.log", "prod_stats_log honours GB_STATS_LOG")

# --- coverage from 4e18 ignores test ranges below it (gstats.py, goldbach_live.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gstats  # noqa: E402
import goldbach_live  # noqa: E402
E = 4000000000000000000
for name, fn in (("gstats", gstats.covered_to), ("goldbach_live", goldbach_live.covered_to)):
    check(fn([(10**12, 10**12 + 10**7), (E, E + 10**7)]) == E + 10**7, f"{name}.covered_to skips a range below 4e18")
    check(fn([(E, E + 10), (E + 12, E + 20), (E + 40, E + 50)]) == E + 20, f"{name}.covered_to stops at a gap")
    check(fn([(E + 2, E + 10)]) is None, f"{name}.covered_to: not starting at 4e18 -> None")

print(f"test_ci: {fails} failure(s) -> {'PASS' if not fails else 'FAIL'}")
sys.exit(1 if fails else 0)
