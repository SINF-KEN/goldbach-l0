#!/usr/bin/env python3
"""test_stats_tools.py — GPU-free self-test of scripts/gstats.py and
scripts/goldbach_live.py on synthetic campaign files (ctest stats_tools).

Builds a run directory with a journal, a stats state / log, a live snapshot,
a ridge store and a records file whose contents are known, runs both tools,
and checks what they report: coverage, prime / twin totals, tail counts, the
record sequence, S(p) found vs proven (contiguity from 4e18), the "beaten"
banner only for a CPU-verified p(n) > 9781, ETA rendering.
usage: tests/test_stats_tools.py   (exit 1 on any failure)
"""
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAIL = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        FAIL.append(what)


def run(args, cwd):
    r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def odd_primes(n):
    s = bytearray([1]) * (n + 1)
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [p for p in range(3, n + 1, 2) if s[p]]


def main():
    pr = odd_primes(9781)
    S0 = 4000000000000000000
    with tempfile.TemporaryDirectory() as d:
        dp = pathlib.Path(d)
        # state: [4e18, 4e18 + 4e11] covered; tail counts for 3001 (5) and 4133 (1)
        i3001, i4133 = pr.index(3001), pr.index(4133)
        tail = [""] * (i4133 + 1)
        tail[i3001], tail[i4133] = "5", "1"
        samp = ["" for _ in range(3)]
        samp[0], samp[1], samp[2] = "70", "20", "10"           # p = 3, 5, 7
        lines = [
            "# synthetic", "tail_t 3000",
            f"range {S0} {S0 + 400000000000}",
            "evens 200000000001 100", "primes30 1 2 3 4 5 6 7 8", "twins30 10 20 30",
            f"first 3001 {S0 + 1000}", f"first 4133 {S0 + 169717704}",
            f"first 8423 {S0 + 300000000000}",                  # unknown-S(p) prime, inside coverage
            "sampled 0 " + ",".join(samp),
            "tailcls 0 " + ",".join(tail), "tailagg ",
        ]
        (dp / "goldbach_stats.state").write_text("\n".join(lines) + "\n")
        (dp / "goldbach_stats.log").write_text(
            f"GBSTATS v2 version=2.2.0 start={S0} limit={S0 + 400000000000} evens=200000000001 "
            "evens_sampled=100 tail_t=3000 tail_evens=6 pmax=4133 primes=36 twins=60 "
            "primes30=1,2,3,4,5,6,7,8 twins30=10,20,30 new_first=- beyond=- tail=3001:5\n")
        j = dp / f"range-{S0}-{S0 + 2000000000000}.journal"
        j.write_text(f"PASS {S0} {S0 + 399999999998} 2026-10-01T00:00:00+02:00 version=2.2.0 "
                     "result=PASS evens_per_sec=5.9e+10\n")
        (dp / "goldbach_ridges.csv").write_text(
            "# store\n1,4000000000169717704,4133,4000000000169713571,4000000000000000000\n")
        # a p(n) = 9787 event OUTSIDE coverage (not proven) and a MISMATCH that must not count
        (dp / "goldbach_records.txt").write_text(
            f"2026-10-01T00:00:00 p(n)=9787 n={S0 + 900000000000} q=1 cpu_check=MISMATCH  test\n"
            f"2026-10-01T00:00:00 p(n)=8429 n={S0 + 1500000000000} q=1 cpu_check=OK  S(p) candidate\n")
        import json
        import time
        (dp / "goldbach_live.json").write_text(json.dumps({
            "time": int(time.time()), "start": S0 + 400000000000, "limit": S0 + 599999999998,
            "evens_done": 50000000000, "evens_total": 100000000000, "elapsed_s": 850.0, "tail_t": 3000,
            "tail_evens": 7, "pmax": 6073, "pmax_n": str(S0 + 402622391632), "ge8000": 0, "ge9000": 0,
            "gt9781": 0, "primes": 1000, "twins": 100, "events": []}))

        print("gstats.py")
        rc, out = run([str(ROOT / "scripts/gstats.py"), "goldbach_stats.state", "--log", "goldbach_stats.log",
                       "--runs"], d)
        check(rc == 0, "gstats.py exits 0")
        check(re.search(r"primes\s+36\s", out) is not None, "prime total 36")
        check(re.search(r"twin pairs\s+60\s", out) is not None, "twin total 60")
        check("evens: 6 " in out, "tail evens = 5 + 1")
        check("p(n) = 4133" in out and f"n = {S0 + 169717704}" in out, "record sequence lists 4133 at its n")
        check("1 runs" in out, "per-run log parsed")
        # --ref / --export on a synthetic t0 (format of T. Oliveira e Silva's table):
        # 8423 unknown ('?') -> NEW S(8423) (state first occurrence inside coverage);
        # record-holders re-marked on the extended table
        (dp / "t0.txt").write_text(
            "pi(p)    p                 S(p)               L(p) finder\n"
            " 431 3001          1390749343154     13585990116 tos\n"
            " 568 4133     500000000000000000*              7 tos\n"
            "1053 8423                    ?*                 0\n"
            "1206 9781* 3325581707333960528*                1 sp\n")
        rc, out = run([str(ROOT / "scripts/gstats.py"), "goldbach_stats.state", "--ref", "t0.txt",
                       "--export", "ext.txt"], d)
        check(rc == 0 and f"NEW  S(8423) = {S0 + 300000000000}" in out, "--ref: S(8423) newly determined")
        ext = (dp / "ext.txt").read_text()
        check(re.search(rf"^\s*8423 +{S0 + 300000000000} NEW +0 +0 +0$", ext, re.M) is not None,
              "--export: NEW S(8423); not a record-holder (S(9781) < S(8423))")
        check(re.search(r"^\s*9781\* +3325581707333960528 ", ext, re.M) is not None,
              "--export: 9781 stays a record-holder")
        check(re.search(r"^\s*4133\* +500000000000000000 +7 +1 +8$", ext, re.M) is not None,
              "--export: L(p) to 4e18 + range = total")

        print("goldbach_live.py")
        rc, out = run([str(ROOT / "scripts/goldbach_live.py"), "--once"], d)
        check(rc == 0, "goldbach_live.py exits 0")
        check(re.search(r"campaign largest p\(n\): (\x1b\[[0-9;]*m)*8423(\x1b\[[0-9;]*m)* at n = 4000000300000000000",
                        out) is not None, "largest p(n) from the state's first occurrences")
        check(re.search(r"\s50\.0%", out) is not None, "current slice 50% from the live snapshot")
        check("found here: \x1b[1m2\x1b[0m (1 proven" in out,
              "S(p): 8423 (covered) proven, 8429 (beyond coverage) found but not proven")
        check("BEATEN" not in out, "a MISMATCH record is never shown as beating 9781")
        check("campaign ETA" in out and "slice" in out, "progress / ETA lines")
        check("4133" in out, "ridge store rendered")
        # contiguity: drop the coverage start -> no S(p) is proven
        st = (dp / "goldbach_stats.state").read_text().replace(f"range {S0} ", f"range {S0 + 2} ")
        (dp / "goldbach_stats.state").write_text(st)
        rc, out = run([str(ROOT / "scripts/goldbach_live.py"), "--once"], d)
        check("(0 proven" in out, "a gap at 4e18 makes S(p) unproven")
        # a verified p(n) > 9781 shows the banner
        with open(dp / "goldbach_records.txt", "a") as f:
            f.write(f"2026-10-01T00:00:00 p(n)=9811 n={S0 + 77} q=1 cpu_check=OK  NEW RECORD\n")
        rc, out = run([str(ROOT / "scripts/goldbach_live.py"), "--once"], d)
        check("BEATEN: p(n) = 9811" in out, "verified p(n) > 9781 -> BEATEN banner")
    print(f"test_stats_tools: {len(FAIL)} failure(s) -> {'FAIL' if FAIL else 'PASS'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
