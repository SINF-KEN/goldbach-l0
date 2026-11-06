#!/usr/bin/env python3
"""known_windows.py — goldbach_l0 against published results from an
independent verifier.

Reference: I. Llorente-Saguer, "Fail-Closed GPU Verification of the Even
Goldbach Conjecture beyond 4e18: Method and Evidence" (GoldbachGPU v3.2.0,
CUDA, RTX 5090; 7 October 2026; code and logs: doi:10.5281/zenodo.23127595).
Its windows [4e18, 4e18 + D] (Table 5) were each run twice with different
segmentations and their prime counts equal primecount (Table 6); the window
at 2^63 is its Table 17, the 1e10 count its Appendix E. A different
program on different hardware with a different sieve, so agreement here is
an external check of the dense path, not a self-consistency test.

For every window: primes in (S, N] (S, N even: the same as goldbach_l0's
`primes`), the largest p(n) of the window and the smallest n attaining it.

usage:
  known_windows.py selftest
        GPU-free (ctest `known_windows`): the reference table is consistent
        (prime counts = differences of the published pi(x) values, every
        listed n has the listed p(n) by Miller-Rabin) and the campaign check
        works on synthetic files.
  known_windows.py run BIN [--upto D] [--only NAME,..] [ARGS...]
        runs BIN on each window of width <= D (default 1e12) with fresh,
        temporary statistics files and compares. Extra ARGS go to BIN
        (e.g. --prime-cache P). Called by tests/gpu_validate.sh.
        Wall time on the B580 (6.4e10 evens/s): 1e12 ~8 s, 1e13 ~80 s,
        1e14 ~13 min, 1e15 ~2.2 h.
  known_windows.py campaign [--state goldbach_stats.state] [--log goldbach_stats.log]
        checks the production campaign's own files: p(n) maxima and their n
        from the state's first occurrences once the campaign covers a window
        without a gap; prime counts where the log's slices tile it. A window
        not covered yet is reported, not failed.
Exit 0 = every check made agreed; 1 = a difference (or a failed run).
"""
import os
import subprocess
import sys
import tempfile

E18 = 4 * 10**18

# name, S, N, primes in (S, N], max p(n) in [S, N] (None: not published),
# smallest n with that p(n), source
WINDOWS = [
    ("4e18+1e10", E18, E18 + 10**10, 233462722, None, None, "App. E"),
    ("4e18+1e12", E18, E18 + 10**12, 23346564662, 6073, E18 + 402622391632, "Table 5"),
    ("4e18+1e13", E18, E18 + 10**13, 233465239142, 6421, E18 + 5756560737472, "Table 5"),
    ("4e18+1e14", E18, E18 + 10**14, 2334657870388, 7103, E18 + 48374322888936, "Table 5"),
    ("4e18+1e15", E18, E18 + 10**15, 23346512560823, 7487, E18 + 400837678526154, "Table 5"),
    ("2^63", 9223371536854775808, 9223372536854775808, 22899962096, 6661, 9223372067978243558,
     "Table 17"),
]
# The paper's other boundary window ends at 2^64 - 2^33, above goldbach_l0's
# supported limit (2^64 - 2^40), so it is not used.

# pi(x) from primecount 8.8 as published (Tables 6 and 17)
PI = {
    E18: 95676260903887607,
    E18 + 10**12: 95676284250452269,
    E18 + 10**13: 95676494369126749,
    E18 + 10**14: 95678595561757995,
    E18 + 10**15: 95699607416448430,
    9223371536854775808: 216289600403451963,
    9223372536854775808: 216289623303414059,
}

MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)   # exact below 3.3e24


def is_prime(n):
    if n < 2:
        return False
    for p in MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def p_of(n):
    """Smallest prime p with n - p prime (n even, n > 4)."""
    p = 3
    while not (is_prime(p) and is_prime(n - p)):
        p += 2
    return p


def kv(tokens):
    return dict(t.split("=", 1) for t in tokens if "=" in t)


def firsts(s):
    """new_first field 'p:n;p:n' -> {p: n}."""
    return {} if s in ("", "-") else {int(p): int(n) for p, n in (e.split(":") for e in s.split(";"))}


def fmt(x):
    return "-" if x is None else f"{x:,}"


# ------------------------------------------------------------------- run --
def run(argv):
    upto, only, extra, i = 10**12, None, [], 1
    while i < len(argv):
        if argv[i] == "--upto":
            upto, i = int(float(argv[i + 1])), i + 2
        elif argv[i] == "--only":
            only, i = set(argv[i + 1].split(",")), i + 2
        else:
            extra.append(argv[i])
            i += 1
    binary, bad, done = argv[0], 0, 0
    for name, S, N, primes, pmax, n_at, src in WINDOWS:
        if (only and name not in only) or (not only and N - S > upto):
            continue
        done += 1
        with tempfile.TemporaryDirectory() as t:
            env = dict(os.environ, GB_STATS_STATE=f"{t}/st.state", GB_STATS_LOG=f"{t}/st.log",
                       GB_RIDGE_FILE=f"{t}/ridges.csv", GB_RECORDS_FILE=f"{t}/records.txt",
                       GB_STATS_LIVE=f"{t}/live.json", GB_RECORD_HOOK="")
            env.pop("GB_STATS", None)
            p = subprocess.run([binary, "--start", str(S), "--limit", str(N), "--csv", *extra],
                               env=env, capture_output=True, text=True)
            out = p.stdout + p.stderr
            log = ""
            if os.path.exists(f"{t}/st.log"):
                log = [ln for ln in open(f"{t}/st.log") if ln.startswith("GBSTATS")]
                log = log[-1] if log else ""
        if p.returncode != 0 or "result=PASS" not in out or not log:
            print(f"  FAIL {name}: run failed (rc {p.returncode})")
            print("       " + "\n       ".join(out.strip().splitlines()[-3:]))
            bad += 1
            continue
        g = kv(log.split()[2:])
        got_n = firsts(g.get("new_first", "-")).get(int(g["pmax"]))
        diffs = []
        if (int(g["start"]), int(g["limit"])) != (S, N):
            diffs.append(f"range [{g['start']}, {g['limit']}]")
        if int(g["primes"]) != primes:
            diffs.append(f"primes {int(g['primes']):,} != {primes:,}")
        if pmax is not None and int(g["pmax"]) != pmax:
            diffs.append(f"max p(n) {g['pmax']} != {pmax}")
        if n_at is not None and got_n != n_at:
            diffs.append(f"n of max p(n) {got_n} != {n_at}")
        if diffs:
            print(f"  FAIL {name} ({src}): " + "; ".join(diffs))
            bad += 1
        else:
            print(f"  ok   {name} ({src}): primes {primes:,}, max p(n) {fmt(pmax)} at n = {fmt(n_at)}")
    if not done:
        print("  no window selected")
        return 1
    return 1 if bad else 0


# -------------------------------------------------------------- campaign --
def read_state(path):
    st = {"range": [], "first": {}, "tail_t": None}
    for line in open(path):
        k, _, rest = line.rstrip("\n").partition(" ")
        if k == "range":
            st["range"].append(tuple(map(int, rest.split())))
        elif k == "first":
            p, n = rest.split()
            st["first"][int(p)] = int(n)
        elif k == "tail_t":
            st["tail_t"] = int(rest)
    return st


def read_log(path):
    runs = set()
    for line in open(path):
        if line.startswith("GBSTATS"):
            d = kv(line.split()[2:])
            runs.add((int(d["start"]), int(d["limit"]), int(d["primes"]), int(d["pmax"])))
    return sorted(runs)


def covered_to(ranges, S):
    """Largest even c such that [S, c] is covered by the union of ranges
    (each [a, b] of evens; adjacent when a = b + 2), or None."""
    c = None
    for a, b in sorted(ranges):
        if b < S:
            continue
        if c is None:
            if a > S:
                return None
            c = b
        elif a <= c + 2:
            c = max(c, b)
        else:
            break
    return c


def tile_count(runs, S, N):
    """Primes in (S, N] from log slices that tile [S, N] exactly (next start
    = previous limit + 2), adding the odd number between two slices, which
    neither slice counts (a run counts primes in (start, limit)). None if
    the slices do not tile the window."""
    inside = [r for r in runs if S <= r[0] and r[1] <= N]
    if not inside or inside[0][0] != S:
        return None
    total, at = 0, None
    for a, b, primes, *_ in inside:
        if at is not None:
            if a != at + 2:
                return None
            total += is_prime(at + 1)
        total, at = total + primes, b
    return total if at == N else None


def campaign(argv):
    state_path, log_path, i = "goldbach_stats.state", "goldbach_stats.log", 0
    while i < len(argv):
        if argv[i] == "--state":
            state_path, i = argv[i + 1], i + 2
        elif argv[i] == "--log":
            log_path, i = argv[i + 1], i + 2
        else:
            print(f"unknown argument {argv[i]}")
            return 2
    st = read_state(state_path)
    runs = read_log(log_path) if os.path.exists(log_path) else []
    T = st["tail_t"] or 3000
    bad = checked = 0
    for name, S, N, primes, pmax, n_at, src in WINDOWS:
        c = covered_to(st["range"], S)
        if c is None or c < N:
            print(f"  --   {name}: not covered yet" + (f" (covered to {c})" if c is not None else ""))
            continue
        notes, diffs = [], []
        if pmax is not None:
            # first[p] = smallest campaign n with p(n) = p, complete for p >= T;
            # with [S, N] covered, the window's max p(n) is the largest p
            # whose first occurrence lies in it
            inw = [(p, n) for p, n in st["first"].items() if p >= T and S <= n <= N]
            gp, gn = max(inw) if inw else (None, None)
            if (gp, gn) != (pmax, n_at):
                diffs.append(f"max p(n) {gp} at {gn} != {pmax} at {n_at}")
            else:
                notes.append(f"max p(n) {pmax} at n = {n_at:,}")
        cnt = tile_count(runs, S, N)
        if cnt is None:
            notes.append("prime count not checked (log slices do not tile the window)")
        elif cnt != primes:
            diffs.append(f"primes {cnt:,} != {primes:,}")
        else:
            notes.append(f"primes {primes:,}")
        checked += 1
        if diffs:
            print(f"  FAIL {name} ({src}): " + "; ".join(diffs))
            bad += 1
        else:
            print(f"  ok   {name} ({src}): " + "; ".join(notes))
    print(f"known_windows campaign: {checked} window(s) checked, {bad} difference(s)")
    return 1 if bad else 0


# -------------------------------------------------------------- selftest --
def selftest():
    fails = []

    def check(ok, what):
        if not ok:
            fails.append(what)

    for name, S, N, primes, pmax, n_at, src in WINDOWS:
        check(S % 2 == 0 and N % 2 == 0 and S < N, f"{name}: even S < N")
        if S in PI and N in PI:
            check(PI[N] - PI[S] == primes, f"{name}: primes == pi(N) - pi(S)")
        if pmax is not None:
            check(S <= n_at <= N and n_at % 2 == 0, f"{name}: n in window")
            check(p_of(n_at) == pmax, f"{name}: p({n_at}) == {pmax}")
    # nested windows: maxima non-decreasing, first n of a larger max beyond
    # the smaller window when the max grows
    nest = [w for w in WINDOWS if w[1] == E18 and w[4] is not None]
    for a, b in zip(nest, nest[1:]):
        check(b[4] >= a[4], f"{b[0]}: max p(n) >= {a[0]}'s")
        if b[4] > a[4]:
            check(b[5] > a[2], f"{b[0]}: a larger max lies outside {a[0]}")

    # campaign mode on synthetic files: a campaign over [4e18, 4e18 + 1e12]
    # in three slices whose boundaries' odd numbers include a prime
    S, N = E18, E18 + 10**12
    b1 = next(b for b in range(S + 2 * 10**11, S + 3 * 10**11, 2) if is_prime(b + 1))
    b2 = S + 6 * 10**11
    gap = is_prime(b1 + 1) + is_prime(b2 + 1)
    with tempfile.TemporaryDirectory() as t:
        def files(ranges, slices, first):
            with open(f"{t}/s", "w") as f:
                f.write("tail_t 3000\n")
                for a, b in ranges:
                    f.write(f"range {a} {b}\n")
                for p, n in first.items():
                    f.write(f"first {p} {n}\n")
            with open(f"{t}/l", "w") as f:
                for a, b, pr in slices:
                    f.write(f"GBSTATS v2 version=x start={a} limit={b} primes={pr} pmax=1\n")
            return ["--state", f"{t}/s", "--log", f"{t}/l"]

        first = {6073: E18 + 402622391632, 5501: E18 + 10**9, 7001: N + 10}
        sl = [(S, b1, 7 * 10**9), (b1 + 2, b2, 9 * 10**9),
              (b2 + 2, N, 23346564662 - 16 * 10**9 - gap)]
        devnull = open(os.devnull, "w")
        old = sys.stdout
        try:
            sys.stdout = devnull
            ok = campaign(files([(S, N)], sl, first))
            wrong_n = campaign(files([(S, N)], sl, {**first, 6073: E18 + 402622391634}))
            wrong_cnt = campaign(files([(S, N)], [sl[0], sl[1], (sl[2][0], N, sl[2][2] + 1)], first))
            short = campaign(files([(S, N - 2)], sl, first))
            hole = campaign(files([(S, b1), (b1 + 4, N)], sl, first))
            no_tile = campaign(files([(S, N)], [sl[0], (b1 + 4, b2, 1), sl[2]], first))
        finally:
            sys.stdout = old
        check(ok == 0, "campaign: matching synthetic campaign passes")
        check(wrong_n == 1, "campaign: wrong n of the max is caught")
        check(wrong_cnt == 1, "campaign: prime count off by one is caught")
        check(short == 0, "campaign: a window not fully covered is skipped")
        check(hole == 0, "campaign: a gap in coverage skips the window")
        check(no_tile == 0, "campaign: slices that do not tile skip the count only")
        check(tile_count(sl, S, N) == 23346564662, "tile_count adds the primes between slices")
        check(covered_to([(S, b1), (b1 + 2, N)], S) == N, "covered_to joins adjacent ranges")
        check(covered_to([(S + 2, N)], S) is None, "covered_to needs S itself")

        # run mode against a stand-in binary that writes a GBSTATS line
        # (FAKE_DELTA shifts its prime count, FAKE_RC its exit status)
        fake = f"{t}/fake.py"
        with open(fake, "w") as f:
            f.write("#!/usr/bin/env python3\nimport os, sys\n"
                    "a = sys.argv[sys.argv.index('--start') + 1]\n"
                    "b = sys.argv[sys.argv.index('--limit') + 1]\n"
                    "with open(os.environ['GB_STATS_LOG'], 'a') as f:\n"
                    f"    f.write(f'GBSTATS v2 version=x start={{a}} limit={{b}} pmax=6073 '\n"
                    f"            f'primes={{23346564662 + int(os.environ.get(\"FAKE_DELTA\", 0))}} '\n"
                    f"            f'new_first=3001:{E18 + 4};6073:{E18 + 402622391632}\\n')\n"
                    "print('result=PASS')\nsys.exit(int(os.environ.get('FAKE_RC', 0)))\n")
        os.chmod(fake, 0o755)
        try:
            sys.stdout = devnull
            r_ok = run([fake, "--only", "4e18+1e12"])
            os.environ["FAKE_DELTA"] = "1"
            r_cnt = run([fake, "--only", "4e18+1e12"])
            os.environ["FAKE_DELTA"] = "0"
            r_max = run([fake, "--only", "4e18+1e13"])
            os.environ["FAKE_RC"] = "1"
            r_rc = run([fake, "--only", "4e18+1e12"])
        finally:
            sys.stdout = old
            os.environ.pop("FAKE_DELTA", None)
            os.environ.pop("FAKE_RC", None)
        check(r_ok == 0, "run: matching output passes")
        check(r_cnt == 1, "run: prime count off by one is caught")
        check(r_max == 1, "run: wrong max p(n) is caught")
        check(r_rc == 1, "run: a non-zero exit fails")
        devnull.close()
    for f in fails:
        print(f"  FAIL: {f}")
    print(f"known_windows selftest: {len(fails)} failure(s) -> {'FAIL' if fails else 'PASS'}")
    return 1 if fails else 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("selftest", "run", "campaign"):
        print(__doc__)
        return 2
    if sys.argv[1] == "selftest":
        return selftest()
    if sys.argv[1] == "run":
        if len(sys.argv) < 3:
            print(__doc__)
            return 2
        return run(sys.argv[2:])
    return campaign(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
