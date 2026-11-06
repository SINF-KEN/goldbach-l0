#!/usr/bin/env python3
"""stats_compare.py LOG STATE CPU_OUT — compare one goldbach_l0 statistics run
(the GBSTATS log line + the state file, written from a fresh state) with the
independent CPU reference (stats_ref START LIMIT). Exit 1 on any difference.

Checked: evens, primes, twins (totals and by residue), the p(n) tail
(p >= tail_t: every even), first occurrences of tail p, the sampled-histogram
total; with every segment sampled (evens_sampled == evens) the full p(n)
histogram bin by bin. Prime gaps: count, sum of gaps, the tail (g >= 700) and
the largest gap exactly; with every segment walked the full gap histogram and
moments too.
Used by tests/gpu_validate.sh.
"""
import sys


def kv(line):
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


def main():
    log, state, cpu = sys.argv[1:4]
    g = kv(open(log).read().strip().splitlines()[-1])
    c = kv(open(cpu).read().strip())
    pr = [p for p in range(3, 9782, 2) if all(p % d for d in range(3, int(p ** 0.5) + 1, 2))]
    T = int(g["tail_t"])
    bad = []
    for k in ("evens", "primes", "twins", "primes30", "twins30"):
        if g.get(k) != c.get(k):
            bad.append(f"{k}: gpu {g.get(k)} cpu {c.get(k)}")
    ch = [int(x) if x else 0 for x in c["hist"].split(",")]
    ch += [0] * (len(pr) - len(ch))
    cpu_tail = {pr[i]: v for i, v in enumerate(ch) if v and pr[i] >= T}
    gt = {}
    if g["tail"] != "-":
        p0, lst = g["tail"].split(":")
        i0 = pr.index(int(p0))
        for i, x in enumerate(lst.split(",")):
            if x:
                gt[pr[i0 + i]] = int(x)
    if gt != cpu_tail:
        bad.append(f"tail differs: {sorted(set(gt.items()) ^ set(cpu_tail.items()))[:6]}")
    acc = {}
    for line in open(state):
        if line.startswith("sampled "):
            _, _, lst = line.rstrip("\n").split(" ", 2)
            for i, x in enumerate(lst.split(",") if lst else []):
                if x:
                    acc[i] = acc.get(i, 0) + int(x)
    gh = [acc.get(i, 0) for i in range(len(pr))]
    if sum(gh) != int(g["evens_sampled"]):
        bad.append(f"sampled histogram total {sum(gh)} != evens_sampled {g['evens_sampled']}")
    if g["evens"] == g["evens_sampled"] and gh != ch[:len(gh)]:
        bad.append("full histogram differs: " +
                   str([(pr[i], a, b) for i, (a, b) in enumerate(zip(gh, ch)) if a != b][:6]))
    gf = {} if g["new_first"] == "-" else {int(a): int(b) for a, b in (x.split(":") for x in g["new_first"].split(";"))}
    cf = {int(a): int(b) for a, b in (x.split(":") for x in c["first"].split(";"))}
    cft = {p: n for p, n in cf.items() if p >= T}
    if gf != cft:
        bad.append(f"first occurrences differ: {sorted(set(gf.items()) ^ set(cft.items()))[:6]}")
    if "gaps" in g:
        # exact on every segment: count, sum of gaps, the tail (g >= 700), the
        # largest gap (when the run reports it exact)
        cm = c.get("gmom", "").split(",")
        for name, gv, cv in (("gaps", g.get("gaps"), c.get("gaps")), ("gd1", g.get("gd1"), cm[0]),
                             ("gtail", g.get("gtail"), c.get("gtail"))):
            if gv != cv:
                bad.append(f"{name}: gpu {gv} cpu {cv}")
        if g.get("gmax") != "-" and g.get("gmax") != c.get("gmax"):
            bad.append(f"gmax: gpu {g.get('gmax')} cpu {c.get('gmax')}")
        # every segment walked (GB_STATS_SAMPLE=1 or GB_GAPS=exact): the walk is the full truth
        we, wn = g.get("gwalk", "0:0").split(":")
        if we == g["evens"]:
            dense = lambda v: [int(x) if x else 0 for x in v.split(",")] if v not in (None, "-") else []
            gs, gh2 = dense(g.get("gsamp")), dense(c.get("ghist"))
            n = max(len(gs), len(gh2))
            gs, gh2 = gs + [0] * (n - len(gs)), gh2 + [0] * (n - len(gh2))
            if gs != gh2:
                bad.append("full gap histogram differs: " + str([(2 * i, a, b) for i, (a, b) in enumerate(zip(gs, gh2)) if a != b][:6]))
            if g.get("gsmom") != c.get("gmom") or wn != c.get("gaps"):
                bad.append(f"walked moments: gpu {wn} {g.get('gsmom')} cpu {c.get('gaps')} {c.get('gmom')}")
            if g.get("gmax") != c.get("gmax"):
                bad.append(f"gmax with every segment walked: gpu {g.get('gmax')} cpu {c.get('gmax')}")
    for b in bad:
        print("  " + b)
    print(("MISMATCH" if bad else "MATCH") +
          f" tail_t={T} tail_evens={g['tail_evens']} sampled={g['evens_sampled']}/{g['evens']} pmax={g['pmax']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
