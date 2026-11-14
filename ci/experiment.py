#!/usr/bin/env python3
"""experiment.py — kernel / knob A/B experiments on the production B580 (stdlib only).

Run by the GitHub workflow `Experiment` (.github/workflows/experiment.yml) on
the goldbach-gpu runner for every push to an `exp/**` branch; the spec is
.github/experiment.json of the pushed commit. Nothing is promoted and the
CI ledger is not touched: this answers "is variant X faster, and is it
still exact?" before a change goes through the normal CI.

Every arm is a (git ref, ocloc -D options, GB_* environment) triple:
  - each distinct ref is built once (cmake Release, goldbach_l0 + stats_ref);
  - each distinct (ref, ocloc options) gets a fresh kernels/goldbach.cl
    compile with the local (pinned) ocloc, so a compile-time variant or a
    timing-only ablation needs no committed .spv;
  - the arm "release": "active" is the running production release as is.
With production paused (the production tree's goldbach_ci.py: slice
boundary, GPU lock, resume on every exit path) it runs
  1. bench.py, every arm interleaved, fresh process per run, production's
     segment and GB_* settings (+ GB_BUCKET_VERIFY=100 as run_range);
  2. an exact check of every arm not marked timing_only: one window with
     production arguments and isolated statistics files, compared field
     by field (csv exact keys + the GBSTATS line) with the active release
     on the same window and with primecount (the CI's own compare());
  3. tests/gpu_validate.sh quick for the arms listed in "gpu_validate".
Report: paired GPU-time ratios vs the reference arm (95% CI), per-phase
medians, the exact checks -> job summary; bench.csv, logs and every
compiled .spv (commit the base arm's module with a kernel change: it is the
pinned-ocloc build the `kernels` gate expects) -> artifact.

Spec (.github/experiment.json):
  {"windows": "long,long-end", "runs": 5, "ref_arm": "base",
   "exact_window": "4000000000000000000:4000000200000000000",
   "gpu_validate": ["few4"],
   "arms": [{"label": "active", "release": "active"},
            {"label": "base"},
            {"label": "few4", "ocloc": "-DLARGE_FEW_F=4"},
            {"label": "gaps", "env": {"GB_GAPS": "sampled"}},
            {"label": "abl", "ocloc": "-DABL_MEDNOSETUP", "timing_only": true},
            {"label": "tile4k", "ocloc": "-DTILE_WORDS=4096",
             "cmake": ["-DCMAKE_CXX_FLAGS=-DGB_TILE_WORDS=4096"]},
            {"label": "main", "ref": "origin/main"}]}
Arm keys: label, ref (default HEAD), release ("active"), ocloc, cmake (list),
env, spv ("shipped": the ref's committed module), timing_only, expect_diff
(results may differ by design, e.g. GB_GAPS=sampled: reported, not failed).
"""
import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # the experiment checkout
T95 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df):
    return T95.get(df) or T95[max(k for k in T95 if k <= df)] if df >= 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--prod-dir", type=Path, required=True, help="production goldbach-l0 tree")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--no-pause", action="store_true", help="do not stop production (debugging only)")
    a = ap.parse_args()

    sys.path.insert(0, str(a.prod_dir / "ci"))
    import goldbach_ci as g                              # the production tree's copy

    spec = json.loads(a.spec.read_text())
    out = a.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    (out / "spv").mkdir(parents=True)
    logf = out / "experiment.log"
    md = []

    def note(s=""):
        md.append(s)
        print(s, flush=True)

    arms = spec["arms"]
    labels = [x["label"] for x in arms]
    if len(set(labels)) != len(labels) or any(":" in s or "," in s or "@" in s for s in labels):
        sys.exit("arm labels must be unique and free of ':' ',' '@'")
    ref_arm = spec.get("ref_arm", labels[0])
    pc = g.prod_conf()
    seg = str(pc.get("env", {}).get("GB_RANGE_SEGMENT", "467500000"))
    base_env = {k: v for k, v in pc.get("env", {}).items()
                if k.startswith("GB_") and not k.startswith("GB_RANGE_") and k not in g.NOTIFY_ENV}
    base_env.setdefault("GB_BUCKET_VERIFY", "100")       # as run_range.sh
    act = g.active_release()
    ameta = g.release_meta(act)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()

    # ---- builds -----------------------------------------------------------
    ocloc = shutil.which("ocloc")
    ocv = subprocess.run([ocloc, "--version"], capture_output=True, text=True).stdout.strip() if ocloc else "-"
    trees = {}

    srcs = {}

    def tree(ref, cmake_args=()):
        """Source tree of ref (HEAD = this checkout), built once per cmake_args."""
        key = (ref, tuple(cmake_args))
        if key in trees:
            return trees[key]
        if ref in ("", "HEAD"):
            src = HERE
        elif ref in srcs:
            src = srcs[ref]
        else:
            src = srcs[ref] = out / f"src-{len(srcs)}"
            g.sh(["git", "-C", HERE, "worktree", "add", "--detach", src, ref], logf=logf, check=True)
        for f in list(a.prod_dir.glob("base_primes*.cache*")):
            if not (src / f.name).exists():
                os.symlink(f, src / f.name)
        bdir = src / f"build-exp{len(trees)}"
        g.sh(["nice", "-n", "15", "cmake", "-S", src, "-B", bdir, "-DCMAKE_BUILD_TYPE=Release", *cmake_args],
             logf=logf, check=True)
        g.sh(["nice", "-n", "15", "cmake", "--build", bdir, "-j", str(os.cpu_count() or 4),
              "--target", "goldbach_l0", "stats_ref"], logf=logf, check=True, timeout=3600)
        sha = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        trees[key] = {"src": src, "build": bdir, "sha": sha}
        return trees[key]

    staged = {}
    for x in arms:
        lb = x["label"]
        d = out / "arms" / lb
        (d / "kernels").mkdir(parents=True)
        if x.get("release") == "active":
            shutil.copy2(act / "goldbach_l0", d / "goldbach_l0")
            shutil.copy2(act / "kernels/goldbach_bmg.spv", d / "kernels/goldbach_bmg.spv")
            staged[lb] = {"dir": d, "what": f"active release {ameta.get('id')} (v{ameta.get('version')})",
                          "src": None}
            continue
        t = tree(x.get("ref", "HEAD"), x.get("cmake", []))
        shutil.copy2(t["build"] / "goldbach_l0", d / "goldbach_l0")
        shutil.copy2(t["build"] / "stats_ref", d / "stats_ref")
        opts = x.get("ocloc", "")
        if x.get("spv") == "shipped":
            shutil.copy2(t["src"] / "kernels/goldbach_bmg.spv", d / "kernels/goldbach_bmg.spv")
        else:
            if not ocloc:
                sys.exit("ocloc not found")
            cmd = ["nice", "-n", "10", ocloc, "compile", "-file", t["src"] / "kernels/goldbach.cl",
                   "-device", "bmg", "-output", out / "spv" / lb] + (["-options", opts] if opts else [])
            rc, o = g.sh(cmd, cwd=out / "spv", logf=out / "spv" / f"{lb}.log", timeout=1800)
            spv = out / "spv" / f"{lb}_bmg.spv"
            if rc != 0 or not spv.exists():
                sys.exit(f"ocloc failed for arm {lb}:\n{o[-3000:]}")
            shutil.copy2(spv, d / "kernels/goldbach_bmg.spv")
        fresh = ""
        if not opts and x.get("spv") != "shipped":
            same = (d / "kernels/goldbach_bmg.spv").read_bytes() == (t["src"] / "kernels/goldbach_bmg.spv").read_bytes()
            fresh = "; fresh compile == shipped .spv" if same else "; fresh compile != shipped .spv (commit spv/" + lb + "_bmg.spv)"
        staged[lb] = {"dir": d, "src": t["src"],
                      "what": f"{x.get('ref', 'HEAD')} {t['sha'][:12]}" + (f", ocloc {opts}" if opts else "")
                              + (f", cmake {' '.join(x['cmake'])}" if x.get("cmake") else "") + fresh}

    note(f"# Experiment {head[:12]}")
    note()
    if spec.get("note"):
        note(spec["note"])
        note()
    note(f"ocloc: `{ocv.splitlines()[0] if ocv else '-'}`; segment {seg}; production env + "
         f"GB_BUCKET_VERIFY={base_env['GB_BUCKET_VERIFY']}; reference arm `{ref_arm}`.")
    note()
    note("| arm | build | env | timing only |")
    note("|---|---|---|---|")
    for x in arms:
        note(f"| `{x['label']}` | {staged[x['label']]['what']} | "
             f"{' '.join(f'{k}={v}' for k, v in x.get('env', {}).items()) or '-'} | "
             f"{'yes' if x.get('timing_only') else ''} |")
    note()

    # ---- GPU: bench, exact windows, validation (production paused) --------
    windows = spec.get("windows", "long,long-end")
    runs = int(spec.get("runs", 5))
    bench_csv = out / "bench.csv"
    exact = {}
    gv = {}
    state = {"run_id": f"experiment-{head[:8]}", "sha": head, "refs": ["experiment"],
             "subject": f"experiment {head[:12]}", "started": time.time()}
    t0 = time.monotonic()
    with g.heartbeat(60), g.flock(g.STATE / "ci.lock", blocking=True, timeout=6 * 3600):
        try:
            with contextlib.ExitStack() as es:
                if not a.no_pause:
                    es.enter_context(g.production_paused(f"experiment {head[:12]}", state))
                es.enter_context(g.flock(g.GPU_LOCK, blocking=True, timeout=600))
                g.set_phase(state, "benchmark", bench_csv=str(bench_csv),
                            bench_plan=[[w, g.window_span(w), runs * len(arms), len(arms)] for w in windows.split(",")],
                            live="")
                cmd = [sys.executable, a.prod_dir / "tools/bench.py", "--windows", windows, "--runs", str(runs),
                       "--warmup", str(spec.get("warmup", 1)), "--segment", seg, "--split", "0",
                       "--ref", ref_arm, "--out", bench_csv]
                for x in arms:
                    env = ",".join(f"{k}={v}" for k, v in x.get("env", {}).items())
                    d = staged[x["label"]]["dir"]
                    cmd += ["--arm", f"{x['label']}{':' + env if env else ''}:@{d / 'goldbach_l0'}@{d / 'kernels/goldbach_bmg.spv'}"]
                rc, _ = g.sh(cmd, cwd=a.prod_dir, env=g.isolated_env(out / "bench-files", base_env),
                             logf=out / "bench.log", timeout=6 * 3600)
                if rc != 0:
                    note(f"**bench.py exited {rc}** (see bench.log)")

                ew = spec.get("exact_window", "4000000000000000000:4000000200000000000")
                if ew:
                    ea, eb = (int(v) for v in ew.split(":"))
                    g.set_phase(state, "experiment_exact")
                    runs_x = {}
                    for x in arms:
                        if x.get("timing_only") and x.get("release") != "active":
                            continue
                        env = dict(base_env, **x.get("env", {}))
                        runs_x[x["label"]] = g.run_slice(staged[x["label"]]["dir"], ea, eb, seg, out,
                                                        env, 3600, name=f"exact-{x['label']}")
                    try:
                        pcount = g.primes_in(ea, eb)
                    except Exception as e:
                        pcount = None
                        note(f"primecount unavailable: {e}")
                    refx = next((runs_x[x["label"]] for x in arms if x.get("release") == "active"), None)
                    for lb, r in runs_x.items():
                        if refx is None:
                            exact[lb] = [{"name": "reference", "level": "warn", "detail": "no active arm"}]
                            continue
                        exact[lb] = g.compare(r, {"kv": refx["csv"], "stats_line": refx["stats_line"]}, pcount)
                    exact["_window"] = f"[{ea}, {eb}]"

                for lb in spec.get("gpu_validate", []):
                    d, src = staged[lb]["dir"], staged[lb]["src"]
                    if src is None:
                        continue
                    g.set_phase(state, "gpu_validate")
                    # gpu_validate.sh runs from the source tree and uses its
                    # kernels/goldbach_bmg.spv: point it at this arm's module
                    bdir = out / f"gv-{lb}"
                    (bdir / "kernels").mkdir(parents=True)
                    for f in ("goldbach_l0", "stats_ref"):
                        shutil.copy2(d / f, bdir / f)
                    shipped = src / "kernels/goldbach_bmg.spv"
                    keep = shipped.read_bytes()
                    shutil.copy2(d / "kernels/goldbach_bmg.spv", shipped)
                    try:
                        rc, o = g.sh(["bash", src / "tests/gpu_validate.sh", bdir, "quick"], cwd=src,
                                     logf=out / f"gpu_validate-{lb}.log", timeout=3600)
                    finally:
                        shipped.write_bytes(keep)
                    gv[lb] = (rc, (o.strip().splitlines() or ["?"])[-1])
        finally:
            g.save_json(g.STATE / "state.json", {"phase": "idle", "last_run": state["run_id"]})
    paused = int(time.monotonic() - t0)

    # ---- report -----------------------------------------------------------
    rows = list(csv.DictReader(open(bench_csv))) if bench_csv.exists() else []
    note(f"## Speed (bench.py, {runs} interleaved runs per arm and window; GPU paused {paused} s)")
    note()
    note("Paired ratio = geometric mean over rounds of ref gpu_s / arm gpu_s (> 1: arm faster), 95% CI.")
    note()
    for w in windows.split(","):
        by = {}
        for r in rows:
            if r.get("window") == w and r.get("status") == "pass":
                by.setdefault(r["label"], {})[r["run"]] = r
        note(f"### {w}")
        note()
        note("| arm | n | G evens/s | speed vs ref | 95% CI | sieve ms | large ms | check ms | sync ms | gpu ms |")
        note("|---|---|---|---|---|---|---|---|---|---|")
        for x in arms:
            lb = x["label"]
            mine = by.get(lb, {})
            if not mine:
                note(f"| `{lb}` | 0 | - | - | - | - | - | - | - | - |")
                continue
            med = lambda k: st.median(float(v[k]) for v in mine.values() if v.get(k) not in (None, ""))
            common = sorted(set(mine) & set(by.get(ref_arm, {})))
            if lb != ref_arm and len(common) >= 2:
                lr = [math.log(float(by[ref_arm][k]["gpu_s"]) / float(mine[k]["gpu_s"])) for k in common]
                gm = math.exp(st.mean(lr))
                hw = t95(len(lr) - 1) * st.stdev(lr) / len(lr) ** 0.5
                ratio, ci = f"{100 * (gm - 1):+.2f}%", f"±{100 * (math.exp(hw) - 1):.2f}%"
            else:
                ratio, ci = ("ref", "") if lb == ref_arm else ("-", "")
            ms = lambda k: f"{1000 * med(k):.1f}" if any(v.get(k) for v in mine.values()) else "-"
            note(f"| `{lb}` | {len(mine)} | {med('evens_per_sec') / 1e9:.3f} | {ratio} | {ci} | "
                 f"{ms('sieve_s')} | {ms('large_s')} | {ms('check_s')} | {ms('sync_s')} | {ms('gpu_s')} |")
        note()
    if exact:
        note(f"## Exact check (window {exact.pop('_window', '?')}, vs the active release + primecount)")
        note()
        note("| arm | result | detail |")
        note("|---|---|---|")
        for lb, checks in exact.items():
            bad = [c for c in checks if c["level"] != "ok"]
            res = "identical" if not bad else ("DIFFERS" if any(c["level"] == "fail" for c in bad) else "warn")
            note(f"| `{lb}` | {res} | {'; '.join(c['name'] + ': ' + c['detail'] for c in bad) or 'all fields equal, primecount ok'} |")
        note()
    if gv:
        note("## tests/gpu_validate.sh quick")
        note()
        for lb, (rc, last) in gv.items():
            note(f"- `{lb}`: {'PASS' if rc == 0 else f'FAIL rc={rc}'} — {last}")
        note()
    (out / "experiment.md").write_text("\n".join(md) + "\n")
    if a.summary:
        with open(a.summary, "a") as f:
            f.write("\n".join(md) + "\n")
    for src in srcs.values():
        subprocess.run(["git", "-C", HERE, "worktree", "remove", "--force", src], capture_output=True)
    shutil.rmtree(out / "arms", ignore_errors=True)
    for p in out.glob("gv-*"):
        shutil.rmtree(p, ignore_errors=True)
    failed = [lb for lb, c in exact.items() if any(x["level"] == "fail" for x in c)
              and not next(x for x in arms if x["label"] == lb).get("expect_diff")]
    failed += [lb for lb, (rc, _) in gv.items() if rc != 0]
    if failed:
        print(f"::error title=Experiment::arms not exact / failed validation: {', '.join(failed)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
