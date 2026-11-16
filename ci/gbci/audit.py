"""Banked slices re-verified: count_check (GPU prime counts vs primecount,
in blocks) and recheck (re-run a slice, compare with its journal row)."""
import contextlib
import json
import time
from pathlib import Path

from .core import GPU_LOCK, STATE, cfg, flock, log, now_iso, read_lines, save_json, stamp
from .gates import compare, run_slice
from .journal import read_journal, stats_line
from .primecount import is_prime, primes_in
from .prodstate import find_run_range, prod_conf, prod_paths, prod_stats_log, save_prod_conf
from .production import production_paused, stop_production
from .releases import active_release, release_dir
from .report import alert


def count_check(journal=None, only_new=True, limit_blocks=None, stop_on_mismatch=False, partial=False):
    """Banked slices' GPU prime counts vs primecount, in blocks of consecutive
    slices covering >= count_block integers (one primecount call per block,
    whatever the slice size); a block that disagrees is re-checked slice by
    slice to find the slice. Results: count-check.jsonl (one row per block,
    plus one per mismatching slice). `partial` also checks a trailing block
    shorter than count_block."""
    c, pc = cfg(), prod_conf()
    j = Path(journal) if journal else prod_paths(pc)[0]
    # resolved = inside a block that matched, or already judged on its own
    # (a known mismatch is handled by the recheck queue and alerted once), or
    # re-verified by `recheck` (RECHECK row in the journal) -- by THIS
    # journal's runs only: another journal over the same integers (a
    # backfill) is a separate GPU run and is checked on its own. Rows before
    # the "journal" field count for slices of the releases they name.
    covered, judged_alone, rechecked = [], {}, set()
    with contextlib.suppress(OSError):
        for r in map(json.loads, read_lines(STATE / "count-check.jsonl")):
            if "journal" in r and r["journal"] != j.name:
                continue
            rels = None if "journal" in r else set(r.get("releases") or [r.get("release", "")])
            if r.get("ok"):
                covered.append((r["a"], r["b"], rels))
            if r.get("slices", 1) == 1:
                judged_alone.setdefault((r["a"], r["b"]), []).append(rels)
    with contextlib.suppress(OSError):
        rechecked = {(int(t[1]), int(t[2])) for t in (ln.split() for ln in read_lines(j)) if t[:1] == ["RECHECK"]}

    def resolved(r):
        mine = lambda rels: rels is None or r["kv"].get("release", "") in rels
        return ((r["a"], r["b"]) in rechecked or any(map(mine, judged_alone.get((r["a"], r["b"]), [])))
                or any(a <= r["a"] and r["b"] <= b and mine(rels) for a, b, rels in covered))
    rows = sorted((r for r in read_journal(j) if r["tag"] == "PASS" and "primes" in r["kv"]
                   and not (only_new and resolved(r))), key=lambda r: r["a"])
    blocks, cur = [], []
    for r in rows:
        if cur and r["a"] != cur[-1]["b"] + 2:
            blocks.append(cur)             # a gap (an unbanked or already checked slice) ends a block
            cur = []
        cur.append(r)
        if cur[-1]["b"] - cur[0]["a"] + 2 >= c["count_block"]:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    blocks = [bl for bl in blocks if partial or bl[-1]["b"] - bl[0]["a"] + 2 >= c["count_block"]
              or bl is not blocks[-1]][:limit_blocks]

    def record(out):
        with open(STATE / "count-check.jsonl", "a") as f:
            f.write(json.dumps(out) + "\n")

    bad = []
    for bl in blocks:
        t0 = time.monotonic()
        a, b = bl[0]["a"], bl[-1]["b"]
        gpu = sum(int(r["kv"]["primes"]) for r in bl)
        # the odd integer between two consecutive slices ([a, b], [b + 2, ...]) is in
        # no slice but in [a, b] of the block: a prime there is not a GPU miss
        between = sum(1 for r in bl[:-1] if is_prime(r["b"] + 1))
        ref = primes_in(a, b) - between
        out = {"a": a, "b": b, "slices": len(bl), "gpu": gpu, "primecount": ref, "between": between, "ok": gpu == ref,
               "releases": sorted({r["kv"].get("release", "") for r in bl}), "journal": j.name, "checked": now_iso(),
               "secs": round(time.monotonic() - t0, 1)}
        record(out)
        print(f"[{a}, {b}] {len(bl)} slice(s): GPU {gpu} primecount {ref} {'OK' if out['ok'] else 'MISMATCH'} "
              f"({out['secs']}s)", flush=True)
        if out["ok"]:
            continue
        for r in bl:                       # localise: slice by slice
            ref_r = primes_in(r["a"], r["b"])
            if str(ref_r) == r["kv"]["primes"]:
                continue
            o = {"a": r["a"], "b": r["b"], "slices": 1, "gpu": int(r["kv"]["primes"]), "primecount": ref_r,
                 "ok": False, "release": r["kv"].get("release", ""), "journal": j.name, "checked": now_iso()}
            record(o)
            bad.append(o)
            log(f"PRIME COUNT MISMATCH in banked slice [{r['a']}, {r['b']}]: GPU {o['gpu']} vs primecount {ref_r}")
            alert("PRIME COUNT MISMATCH", f"slice [{r['a']}, {r['b']}]: GPU {o['gpu']} vs primecount {ref_r} — "
                                          "this PASS is not trustworthy; `goldbach_ci.py recheck` it")
            with open(j.with_suffix(".recheck"), "a") as f:
                f.write(f"{r['a']} {r['b']} {now_iso()} count_mismatch gpu={o['gpu']} primecount={ref_r}\n")
    if bad and stop_on_mismatch and find_run_range():
        stop_production(reason="prime-count mismatch")
        pc = prod_conf()                  # stop_production may have adopted / saved it
        pc["desired"] = "stopped"
        save_prod_conf(pc)
    return len(blocks), bad


# --------------------------------------------------------------- recheck --
def recheck(a=None, release=None):
    """Re-run banked slice(s) — default: the .recheck queue — and compare them
    with their journal rows and primecount; a RECHECK row records the outcome."""
    pc = prod_conf()
    j = prod_paths(pc)[0]
    rows = {(r["a"], r["b"]): r for r in read_journal(j) if r["tag"] == "PASS"}
    if a is not None:
        queue = [k for k in rows if k[0] == int(a)]
    else:
        done = set()
        with contextlib.suppress(OSError):
            done = {tuple(int(x) for x in ln.split()[:2]) for ln in read_lines(STATE / "rechecked.txt")}
        qf = j.with_suffix(".recheck")
        queue = list(dict.fromkeys(k for k in (tuple(int(x) for x in ln.split()[:2])
                                                for ln in (read_lines(qf) if qf.exists() else []))
                                   if k in rows and k not in done))
    if not queue:
        print("nothing to recheck")
        return []
    rel = release_dir(release) if release else active_release()
    seg = pc.get("env", {}).get("GB_RANGE_SEGMENT", "467500000")
    results = []
    with production_paused("recheck"), flock(GPU_LOCK, blocking=True, timeout=120):
        for k in queue:
            ref = rows[k]
            run_dir = STATE / "runs" / f"{stamp()}-recheck-{k[0]}"
            run_dir.mkdir(parents=True)
            log(f"recheck [{k[0]}, {k[1]}] with release {rel.name}")
            res = run_slice(rel, k[0], k[1], seg, run_dir, pc.get("env", {}),
                            max(600, int(3 * float(ref["kv"].get("total_s", 1800)))))
            try:
                count = primes_in(*k)
            except RuntimeError:
                count = None
            checks = compare(res, {"kv": ref["kv"], "stats_line": stats_line(prod_stats_log(pc), *k)}, count)
            ok = not any(x["level"] == "fail" for x in checks)
            save_json(run_dir / "report.json", {"slice": k, "checks": checks, "ok": ok, "csv": res["csv"]})
            with open(STATE / "rechecked.txt", "a") as f:
                f.write(f"{k[0]} {k[1]} {now_iso()} {'CONFIRMED' if ok else 'MISMATCH'} {rel.name}\n")
            with open(j, "a") as f:        # non-PASS row: ignored by the resume chain and dashboards
                f.write(f"RECHECK {k[0]} {k[1]} {now_iso()} outcome={'CONFIRMED' if ok else 'DIFFERS'} "
                        f"release={rel.name} primes={res['csv'].get('primes')} twins={res['csv'].get('twins')} "
                        f"result={res['csv'].get('result')} primecount={count}\n")
            log(f"recheck [{k[0]}, {k[1]}]: {'CONFIRMED' if ok else 'MISMATCH'}")
            if not ok:
                alert("RECHECK MISMATCH", f"slice [{k[0]}, {k[1]}] re-run differs: "
                      + ", ".join(x["name"] for x in checks if x["level"] == "fail"))
            results.append((k, ok))
    return results
