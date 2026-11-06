#!/usr/bin/env python3
"""gstats.py — summarise goldbach-l0's exact Goldbach statistics.

Reads the campaign state (goldbach_stats.state, cumulative) and optionally
the per-run log (goldbach_stats.log, one line per verified slice) written by
the dense verifier (include/gstats.hpp). Everything in those files is exact;
this script only adds the classical heuristic predictions for comparison.

  p(n)  = smallest prime p with n - p prime ("minimal Goldbach prime")
  tail  = every even with p(n) >= tail_t (3000), complete over the range
  head  = the full p(n) histogram of the sampled segments (exact counts for
          exactly those evens; evens_sampled says how many)

usage: scripts/gstats.py [STATE] [--log LOG] [--runs] [--top N]
                         [--ref t0.txt.gz] [--export FILE]

--ref   T. Oliveira e Silva's table to 4e18 (S(p), L(p) per prime p):
        curl -O https://sweet.ua.pt/tos/goldbach/t0.txt.gz
--export writes the same table extended by this campaign (tail primes,
        every even): S(p) over [4, max covered] where the campaign covers
        [4e18, ...] without a gap (NEW where t0 has '?'), L(p) of the range,
        and L(p) to the covered limit = t0 + range. Record-holders re-marked.
"""
import gzip
import argparse
import math
import sys

UNITS = [1, 7, 11, 13, 17, 19, 23, 29]
C2 = 0.6601618158468695739      # twin prime constant


def odd_primes(n):
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\0\0"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [p for p in range(3, n + 1, 2) if s[p]]


PR = odd_primes(9781)


def dense(lst, start=0):
    """'a,b,,c' -> {PR[start + i]: count} (empty = 0)."""
    out = {}
    if not lst or lst == "-":
        return out
    for i, x in enumerate(lst.split(",")):
        if x:
            out[PR[start + i]] = int(x)
    return out


def li(x):
    """Logarithmic integral (Ramanujan's series), good to ~1e-15 relative."""
    lx = math.log(x)
    g = 0.5772156649015329
    s, term, fact, k = 0.0, 1.0, 1.0, 0
    inner = 0.0
    for n in range(1, 200):
        fact *= n
        term = (-1) ** (n - 1) * lx ** n / (fact * 2 ** (n - 1))
        while k <= (n - 1) // 2:
            inner += 1.0 / (2 * k + 1)
            k += 1
        s += term * inner
        if abs(term * inner) < 1e-17 * abs(s):
            break
    return g + math.log(lx) + math.sqrt(x) * s


def twin_hl(a, b, steps=2000):
    """2 C2 * integral_a^b dt / ln(t)^2 (Hardy-Littlewood twin estimate)."""
    h = (b - a) / steps
    f = lambda t: 1.0 / math.log(t) ** 2
    s = f(a) + f(b) + sum((4 if i % 2 else 2) * f(a + i * h) for i in range(1, steps))
    return 2 * C2 * s * h / 3


def read_state(path):
    st = {"range": [], "first": {}, "beyond": [], "sampled": {}, "tailcls": {}, "tailagg": {}}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        k, _, rest = line.rstrip("\n").partition(" ")
        if k == "range":
            st["range"].append(tuple(map(int, rest.split())))
        elif k == "evens":
            st["evens"], st["sampled_evens"] = map(int, rest.split())
        elif k == "tail_t":
            st["tail_t"] = int(rest)
        elif k in ("primes30", "twins30"):
            st[k] = list(map(int, rest.split()))
        elif k == "first":
            p, n = rest.split()
            st["first"][int(p)] = int(n)
        elif k == "beyond":
            p, n = rest.split()
            st["beyond"].append((int(n), int(p)))
        elif k in ("sampled", "tailcls"):
            cr, _, lst = rest.partition(" ")
            st[k][int(cr)] = dense(lst)
        elif k == "tailagg":
            st["tailagg"] = dense(rest)
    return st


def read_log(path):
    runs = []
    for line in open(path):
        if not line.startswith("GBSTATS"):
            continue
        d = dict(t.split("=", 1) for t in line.split()[2:] if "=" in t)
        tail = {}
        if d.get("tail", "-") != "-":
            p0, lst = d["tail"].split(":", 1)
            tail = dense(lst, PR.index(int(p0)))
        d["tail"] = tail
        runs.append(d)
    return runs


def read_t0(path):
    """t0.txt(.gz): {p: (S or None, L)}."""
    op = gzip.open if path.endswith(".gz") else open
    ref = {}
    with op(path, "rt") as f:
        for line in f:
            t = line.split()
            if len(t) >= 4 and t[0].isdigit():
                p = int(t[1].rstrip("*"))
                S = t[2].rstrip("*")
                ref[p] = (None if S == "?" else int(S), int(t[3]))
    return ref


def write_export(path, ref, tail, st, cov, T):
    """t0-style table for p >= T, extended to the campaign's covered limit."""
    rows = []
    for p in sorted(set(ref) | set(tail)):
        if p < T:
            continue
        S_ref, L_ref = ref.get(p, (None, 0))
        f = st["first"].get(p)
        S = S_ref if S_ref is not None else (f if (f and cov is not None and f <= cov) else None)
        new = S_ref is None and S is not None
        L_rng = tail.get(p, 0)
        rows.append([p, S, new, L_ref, L_rng, f])
    # record-holders on the extended S: p is one if S(q) > S(p) for all q > p
    INF = 1 << 64
    m = INF
    for r in reversed(rows):
        Sv = r[1] if r[1] is not None else INF
        r.append(Sv < m and r[1] is not None)
        m = min(m, Sv)
    with open(path, "w") as out:
        out.write("# Extension of T. Oliveira e Silva's goldbach table t0 (verification to 4e18)\n"
                  f"# by goldbach-l0 v2.2 over [4e18, {cov}] (contiguous coverage; exact counts)\n"
                  f"# primes p >= {T} (below that the campaign keeps a sampled histogram only)\n"
                  "# S(p): least even n with p(n) = p over [4, covered limit]; NEW = was '?' in t0\n"
                  "# L4e18: count to 4e18 (t0)   Lrange: count in the campaign range   L: total\n"
                  "# * = record-holder (S(q) > S(p) for every q > p, on the extended table)\n"
                  f"#{'p':>6} {'S(p)':>21} {'':3} {'L4e18':>10} {'Lrange':>10} {'L':>10}\n")
        for p, S, new, L_ref, L_rng, f, rec in rows:
            out.write(f"{p:>6}{'*' if rec else ' '} {(str(S) if S else '?'):>21} {'NEW' if new else '':3} "
                      f"{L_ref:>10} {L_rng:>10} {L_ref + L_rng if cov else '-':>10}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("state", nargs="?", default="goldbach_stats.state")
    ap.add_argument("--log", default=None, help="per-run log (goldbach_stats.log)")
    ap.add_argument("--runs", action="store_true", help="one line per logged run")
    ap.add_argument("--top", type=int, default=15, help="largest p(n) values to list")
    ap.add_argument("--ref", default=None, help="t0.txt(.gz) of T. Oliveira e Silva")
    ap.add_argument("--export", default=None, help="write the extended t0-style table here")
    a = ap.parse_args()
    st = read_state(a.state)
    T = st.get("tail_t", 3000)

    print("== coverage")
    for lo, hi in st["range"]:
        print(f"  [{lo}, {hi}]  ({hi - lo:.4e} integers)")
    ev, se = st.get("evens", 0), st.get("sampled_evens", 0)
    print(f"  evens verified {ev:,}   with full p(n) record (sampled) {se:,} ({100 * se / max(ev, 1):.2f}%)")

    print("\n== primes and twin primes (exact counts vs. classical estimates)")
    p30, t30 = st.get("primes30", [0] * 8), st.get("twins30", [0] * 3)
    np_, nt = sum(p30), sum(t30)
    li_sum = sum(li(hi) - li(lo) for lo, hi in st["range"])
    hl_sum = sum(twin_hl(lo, hi) for lo, hi in st["range"])
    print(f"  primes      {np_:>22,}   Li estimate {li_sum:,.0f}   diff {np_ - li_sum:+,.0f}")
    print(f"  twin pairs  {nt:>22,}   Hardy-Littlewood {hl_sum:,.0f}   ratio {nt / max(hl_sum, 1):.6f}")
    if np_:
        print("  primes by residue mod 30 (prime race; mean = 1/8):")
        for u, c in zip(UNITS, p30):
            print(f"    {u:>2}: {c:>20,}  {c / np_ - 0.125:+.3e}")
        print("  twins by type: " + ", ".join(f"({u}, {u + 2}): {c:,}" for u, c in zip((11, 17, 29), t30)))

    print(f"\n== p(n) tail: every even with p(n) >= {T} (complete)")
    tail = {}
    for cls in st["tailcls"].values():
        for p, c in cls.items():
            tail[p] = tail.get(p, 0) + c
    for p, c in st["tailagg"].items():
        tail[p] = tail.get(p, 0) + c
    ntail = sum(tail.values()) + len(st["beyond"])
    print(f"  evens: {ntail:,}  (1 in {ev / max(ntail, 1):,.0f})")
    if tail:
        top = sorted(tail)[-a.top:][::-1]
        print("  largest p(n) values (count of evens, first n in range):")
        for p in top:
            print(f"    p(n) = {p:>5}: {tail[p]:>8,}   first n = {st['first'].get(p, '?')}")
        for lo_p in (T, 4000, 5000, 6000, 7000, 8000, 9000):
            k = sum(c for p, c in tail.items() if p >= lo_p)
            if k:
                print(f"    p(n) >= {lo_p}: {k:,}")
    for n, p in st["beyond"]:
        print(f"  ** p(n) = {p} > 9781 at n = {n} (new record beyond the candidate list)")

    # records: n where p(n) exceeds every p(m) for m < n in the covered range
    if st["first"]:
        print("\n== p(n) records in range (n ascending, each p(n) larger than all before; tail p only)")
        best = 0
        for p, n in sorted(((p, n) for p, n in st["first"].items()), key=lambda x: x[1]):
            if p > best:
                best = p
                print(f"    n = {n}   p(n) = {p}")

    if st["sampled"]:
        print("\n== p(n) head: full histogram of the sampled segments (exact for those evens)")
        hist = {}
        for cls in st["sampled"].values():
            for p, c in cls.items():
                hist[p] = hist.get(p, 0) + c
        tot = sum(hist.values())
        mean = sum(p * c for p, c in hist.items()) / tot
        acc, q = 0, {}
        for p in sorted(hist):
            acc += hist[p]
            for f in (0.5, 0.9, 0.99, 0.999, 0.999999):
                if f not in q and acc >= f * tot:
                    q[f] = p
        print(f"  evens {tot:,}   mean p(n) {mean:.3f}   " +
              "  ".join(f"q{f:g}={q[f]}" for f in sorted(q)))
        print("  most frequent: " + ", ".join(f"{p}: {100 * hist[p] / tot:.3f}%"
                                              for p in sorted(hist, key=hist.get, reverse=True)[:8]))
        print("  mean p(n) by n mod 30 (singular-series effect):")
        for cr in sorted(st["sampled"]):
            cls = st["sampled"][cr]
            n_ = sum(cls.values())
            if n_:
                m = sum(p * c for p, c in cls.items()) / n_
                print(f"    n = {2 * cr:>2} mod 30: {m:8.3f}   ({n_:,} evens)")

    if a.ref:
        ref = read_t0(a.ref)
        cov = None
        for lo, hi in sorted(st["range"]):
            if cov is None:
                if lo > 4000000000000000000:
                    break
                cov = hi
            elif lo <= cov + 2:
                cov = max(cov, hi)
        tail = {}
        for cls in st["tailcls"].values():
            for p, c in cls.items():
                tail[p] = tail.get(p, 0) + c
        for p, c in st["tailagg"].items():
            tail[p] = tail.get(p, 0) + c
        newS = {p: n for p, n in st["first"].items()
                if p in ref and ref[p][0] is None and cov is not None and n <= cov}
        print(f"\n== vs T. Oliveira e Silva's table to 4e18 ({a.ref})")
        print(f"  campaign covers [4e18, {cov}] without a gap" if cov else
              "  campaign does NOT start at 4e18 (or has a gap there): no S(p) can be claimed")
        unk = [p for p in ref if ref[p][0] is None]
        print(f"  S(p) unknown in t0: {len(unk)} primes; determined by this campaign: {len(newS)}")
        for p in sorted(newS):
            print(f"    NEW  S({p}) = {newS[p]}")
        big = [(p, n) for n, p in st["beyond"] if p > 9781]
        for p, n in big:
            print(f"  *** p(n) = {p} > 9781 at n = {n}: beyond every value known to 4e18 ***")
        if a.export:
            write_export(a.export, ref, tail, st, cov, T)
            print(f"  wrote {a.export}")

    if a.log:
        runs = read_log(a.log)
        print(f"\n== per-run log {a.log}: {len(runs)} runs")
        if a.runs:
            for r in runs:
                print(f"  [{r['start']}, {r['limit']}] pmax={r['pmax']} tail={r['tail_evens']} "
                      f"primes={r['primes']} twins={r['twins']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
