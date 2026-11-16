"""GPU gates and the exact comparison: isolated environments, the bench.py
A/B, one CI-window run, reference-cache keys, compare() (bit for bit and
primecount), the run's figures."""
import csv
import hashlib
import json
import math
import os
import shutil
import statistics as st
import subprocess
import sys
import time

from .core import (BENCH, EXACT_CSV, NOTIFY_ENV, ROOT, STATS_NEW_OK, STATS_STATEFUL, WATERFALL, cfg,
                   csv_header, csv_rows, killpg, kv, log_exc, plot_python, sh)
from .journal import stats_line


def isolated_env(tmp, base_env=None):
    """Environment for CI GPU runs: production's GB_* settings (no
    notifications), statistics / ridge / records / live files under tmp."""
    tmp.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in NOTIFY_ENV}
    env.update({k: v for k, v in (base_env or {}).items() if k not in NOTIFY_ENV})
    env.update({"GB_STATS_STATE": str(tmp / "stats.state"), "GB_STATS_LOG": str(tmp / "stats.log"),
                "GB_RIDGE_FILE": str(tmp / "ridges.csv"), "GB_RECORDS_FILE": str(tmp / "records.txt"),
                "GB_STATS_LIVE": str(tmp / "live.json"), "GB_RECORD_HOOK": ""})
    return env


def run_bench(active, cand, run_dir, seg):
    """bench.py A/B, active vs candidate, interleaved. -> per-window paired GPU-time ratios."""
    c = cfg()
    out = run_dir / "bench.csv"
    cmd = [sys.executable, BENCH, "--windows", c["bench_windows"], "--runs", str(c["bench_runs"]),
           "--warmup", "1", "--out", out, "--segment", str(seg), "--split", "0",
           "--arm", f"active:@{active / 'goldbach_l0'}@{active / 'kernels/goldbach_bmg.spv'}",
           "--arm", f"cand:@{cand / 'goldbach_l0'}@{cand / 'kernels/goldbach_bmg.spv'}"]
    rc, _ = sh(cmd, cwd=ROOT, env=isolated_env(run_dir / "bench-files"), logf=run_dir / "bench.log", timeout=3600)
    res = {"rc": rc, "windows": {}}
    rows = csv_rows(out)
    for w in c["bench_windows"].split(","):
        by = {}
        for r in rows:
            if r.get("window") == w and r.get("status") == "pass":
                by.setdefault(r["label"], {})[r["run"]] = r
        common = sorted(set(by.get("active", {})) & set(by.get("cand", {})))
        if not common:
            res["windows"][w] = {"ok": False}
            continue
        lr = [math.log(float(by["active"][k]["gpu_s"]) / float(by["cand"][k]["gpu_s"])) for k in common]
        g = math.exp(st.mean(lr))
        hw = 2.776 * st.stdev(lr) / len(lr) ** 0.5 if len(lr) > 1 else 0.0
        med = lambda arm: round(st.median(float(by[arm][k]["evens_per_sec"]) for k in common) / 1e9, 3)
        res["windows"][w] = {"ok": True, "n": len(common), "active_gevs": med("active"), "cand_gevs": med("cand"),
                             "speed_ratio": round(g, 5), "ci_pct": round(100 * (math.exp(hw) - 1), 3),
                             "gpu_slowdown_pct": round(100 * (1 / g - 1), 3)}
    return res


def run_slice(rel, a, b, seg, run_dir, base_env, timeout, name="slice"):
    """One goldbach_l0 run of [a, b] with production's arguments + env and
    isolated files. -> {rc, wall_s, csv, line, stats_line, live}."""
    tmp = run_dir / f"{name}-files"
    shutil.rmtree(tmp, ignore_errors=True)
    env = isolated_env(tmp, base_env)
    env.setdefault("GB_BUCKET_VERIFY", "100")            # as run_range.sh
    cmd = [rel / "goldbach_l0", "--start", a, "--limit", b, "--segment", seg, "--split", "0",
           "--spv", rel / "kernels/goldbach_bmg.spv", "--csv"]
    t0 = time.monotonic()
    with open(run_dir / f"{name}.out", "w") as fo, open(run_dir / f"{name}.err", "w") as fe:
        fo.write("$ " + " ".join(map(str, cmd)) + "\n")
        fo.flush()
        p = subprocess.Popen([str(x) for x in cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=fe, text=True, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            killpg(p)
            out, _ = p.communicate()
            rc = 124
        except BaseException:
            killpg(p)
            raise
        fo.write(out)
    line = next((ln for ln in out.splitlines() if "result=" in ln), "")
    return {"rc": rc, "wall_s": round(time.monotonic() - t0, 1), "csv": kv(line), "line": line,
            "stats_line": stats_line(tmp / "stats.log", a, b)}


def ref_key(fp, a, b, seg, env):
    """Cache key of a reference window result: code + window + every GB_*
    setting that can change the outcome."""
    rel_env = {k: v for k, v in sorted(env.items())
               if k.startswith("GB_") and k not in NOTIFY_ENV and not k.startswith("GB_RANGE_")}
    rel_env.setdefault("GB_BUCKET_VERIFY", "100")
    return hashlib.sha256(json.dumps([fp, int(a), int(b), str(seg), rel_env]).encode()).hexdigest()[:20]


def compare(res, ref, prime_count=None):
    """Checks of a CI window result against a reference ({kv, stats_line})."""
    checks = []

    def chk(name, ok, detail, level_fail="fail"):
        checks.append({"name": name, "level": "ok" if ok else level_fail, "detail": detail})

    s, r = res["csv"], ref["kv"]
    chk("slice_exit", res["rc"] == 0 and s.get("result") == "PASS", f"rc={res['rc']} result={s.get('result')}")
    for k in EXACT_CSV:
        if k in r:
            chk(f"exact_{k}", s.get(k) == r.get(k), f"cand {s.get(k)} vs ref {r.get(k)}")
    if ref.get("stats_line") and res["stats_line"]:
        a_, b_ = kv(ref["stats_line"]), kv(res["stats_line"])
        # a field only the candidate writes is new statistics (e.g. the prime
        # gaps of v2.4): nothing to compare yet; a field it drops is a failure
        new = sorted(set(b_) - set(a_))
        diff = sorted(k for k in set(a_) | (set(b_) - STATS_NEW_OK) if k not in STATS_STATEFUL and a_.get(k) != b_.get(k))
        chk("exact_stats_line", not diff, ("identical (all fields but version/new_first)" if not diff
            else f"differs in {diff}") + (f"; new fields {new}" if new else ""))
        if "gaps" in b_:
            # every counted prime has exactly one gap (q - prev(q))
            chk("gaps_invariant", b_.get("gaps") == b_.get("primes") and b_.get("gsmom", "").count(",") == 3,
                f"gaps {b_.get('gaps')} vs primes {b_.get('primes')}")
    else:
        chk("exact_stats_line", False, f"missing stats line (ref {bool(ref.get('stats_line'))}, "
                                       f"cand {bool(res['stats_line'])})")
    if prime_count is None:
        chk("primecount", False, "primecount not available", level_fail="warn")
    else:
        chk("primecount", str(prime_count) == s.get("primes"), f"GPU {s.get('primes')} vs primecount {prime_count}")
    return checks


def ci_figures(run_dir, slices):
    """Waterfall figures (active vs candidate, per window) from this run's own
    data: bench.csv + the CI window runs. Best effort."""
    out = run_dir / "bench.csv"
    figs = []
    try:
        if not out.exists():
            return figs
        hdr = csv_header(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
            for label, res in slices:
                w.writerow({**{k: res["csv"].get(k, "") for k in hdr}, "label": label, "window": "ci-window",
                            "status": "pass" if res["rc"] == 0 else "fail", "run": 1, "wall_s": res["wall_s"],
                            "session": run_dir.name, "detail": f"ci window label={label}"})
        for win in cfg()["bench_windows"].split(",") + (["ci-window"] if slices else []):
            png = run_dir / f"waterfall-{win}.png"
            rc, _ = sh([plot_python(), WATERFALL, out, "--window", win, "--group-by", "label",
                        "--label", "active,cand", "--per", "1e10", "-o", png], logf=run_dir / "figures.log", timeout=300)
            if rc == 0 and png.exists():
                figs.append(png.name)
    except Exception as e:     # figures are cosmetic: no figure must ever change a verdict
        log_exc(f"figures: {type(e).__name__}: {e}")
    return figs
