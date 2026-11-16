"""Benchmark sessions (production paused): arms, history builds, figures
and the per-phase analysis."""
import contextlib
import csv
import shutil
import statistics as st
import sys
import time

from .core import (BENCH, BENCH_PROFILES, GPU_LOCK, ROOT, STATE, WATERFALL, cfg, csv_header, csv_rows, flock,
                   git, log, plot_python, set_phase, sh)
from .gates import isolated_env
from .production import production_paused
from .releases import active_release, link_target, release_meta


PHASES = ["init_s", "sieve_s", "large_s", "check_s", "sync_s", "gpu_s"]


def history_build(label, commit, logf):
    """Build an old commit once (cached in ci-state/bench-builds/)."""
    d = STATE / "bench-builds" / label
    if (d / "build/goldbach_l0").exists():
        return d
    shutil.rmtree(d, ignore_errors=True)
    git("worktree", "prune")
    sh(["git", "-C", ROOT, "worktree", "add", "--detach", d, commit], logf=logf, check=True)
    (d / "tools").unlink(missing_ok=True)
    sh(["nice", "-n", "15", "cmake", "-S", d, "-B", d / "build", "-DCMAKE_BUILD_TYPE=Release"], logf=logf, check=True)
    sh(["nice", "-n", "15", "cmake", "--build", d / "build", "-j", str(cfg()["build_jobs"]), "--target", "goldbach_l0"],
       logf=logf, check=True, timeout=3600)
    return d


def bench_analysis(rows, cur, windows):
    """Markdown: medians per arm and window, and for every other arm the
    per-phase difference that explains a faster or slower GPU time."""
    L = []
    for w in windows:
        by = {}
        for r in rows:
            if r.get("window") == w and r.get("status") == "pass":
                by.setdefault(r["label"], []).append(r)
        if cur not in by:
            continue
        med = {a: {k: st.median(float(x[k]) for x in rs if x.get(k)) for k in PHASES + ["evens_per_sec"]}
               for a, rs in by.items()}
        c = med[cur]
        L += [f"### {w}", "", "| arm | G evens/s | GPU ms | init | sieve | large | check | sync | GPU time vs current |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a, m in sorted(med.items(), key=lambda x: -x[1]["evens_per_sec"]):
            L.append(f"| {a} | {m['evens_per_sec'] / 1e9:.3f} | {1e3 * m['gpu_s']:.1f} | "
                     + " | ".join(f"{1e3 * m[k]:.1f}" for k in PHASES[:-1])
                     + f" | {100 * (m['gpu_s'] / c['gpu_s'] - 1):+.2f}% |")
        L.append("")
        for a, m in med.items():
            if a == cur:
                continue
            dg = c["gpu_s"] - m["gpu_s"]
            if abs(dg) / m["gpu_s"] < 0.002:
                L.append(f"- {cur} vs {a}: no meaningful difference ({100 * dg / m['gpu_s']:+.2f}% GPU time)")
                continue
            parts = sorted(((c[k] - m[k], k) for k in ("sieve_s", "large_s", "check_s", "sync_s")),
                           key=lambda x: -abs(x[0]))
            L.append(f"- {cur} is {'SLOWER' if dg > 0 else 'faster'} than {a} by {100 * dg / m['gpu_s']:+.2f}% "
                     f"GPU time ({1e3 * dg:+.1f} ms): "
                     + ", ".join(f"{k[:-2]} {1e3 * v:+.1f} ms" for v, k in parts[:3]))
        L.append("")
    return L


def benchmark_session(out_dir, profile="full", pause=True, state=None):
    """bench.py session with production paused: current release (+ production
    config, + GB_STATS=0), previous release, history, sparse — per profile.
    Rows -> bench/bench-results.csv; figures + benchmark.md (analysis) -> out_dir."""
    c, P = cfg(), BENCH_PROFILES[profile]
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = out_dir / "benchmark.log"
    act, prev = active_release(), link_target("previous")
    cur = "v" + release_meta(act).get("version", "?")
    arms = {"cur": (cur, {}, act), "prod": (cur + "-prod", {"GB_BUCKET_VERIFY": "100"}, act),
            "nostats": (cur + "-nostats", {"GB_STATS": "0"}, act)}
    if prev:
        pv = "v" + release_meta(prev).get("version", "?")
        arms["prev"] = (pv + ("-prev" if pv == cur else ""), {}, prev)
    sel = [arms[k] for k in P["arms"] if k in arms]
    if P["history"]:
        for label, commit in c["bench_history"].items():
            try:
                sel.append((label, {}, history_build(label, commit, logf)))
            except (RuntimeError, OSError) as e:     # sh / git (check=True), worktree files
                log(f"benchmark: history build {label} failed: {e}")

    def arm(label, env, d):
        b = d / "goldbach_l0" if (d / "goldbach_l0").exists() else d / "build/goldbach_l0"
        e = (":" + ",".join(f"{k}={v}" for k, v in env.items())) if env else ""
        return ["--arm", f"{label}{e}:@{b}@{d / 'kernels/goldbach_bmg.spv'}"]

    csv_out = out_dir / "bench.csv"
    env = isolated_env(out_dir / "files")
    t0 = time.monotonic()
    n_a = len(sel) + (1 if P["sparse"] else 0)
    n_b = len([x for x in sel if not x[0].endswith("-nostats")])
    plan = [[w, window_span(w), P["runs"] * n_a, n_a] for w in P["windows"].split(",")] + \
           [[w.split("=")[0], window_span(w), P["big_runs"] * n_b, n_b] for w in P["big"].split(",")]
    ctx = production_paused("benchmark", state) if pause else contextlib.nullcontext({})
    with ctx, flock(GPU_LOCK, blocking=True, timeout=600):
        if state is not None:
            set_phase(state, "benchmark", bench_csv=str(csv_out), bench_plan=plan, live="")
        extra = ["--arm", f"{cur}-sparse:GB_SPARSE=1:@{act / 'goldbach_l0'}"] if P["sparse"] else []
        sh([sys.executable, BENCH, *sum((arm(*x) for x in sel), []), *extra, "--windows", P["windows"],
            "--runs", str(P["runs"]), "--warmup", "1", "--split", "0", "--out", csv_out],
           cwd=ROOT, env=env, logf=logf, timeout=4 * 3600)
        big = [x for x in sel if not x[0].endswith("-nostats")]
        sh([sys.executable, BENCH, *sum((arm(*x) for x in big), []), "--windows", P["big"], "--runs",
            str(P["big_runs"]), "--warmup", "1", "--split", "0", "--out", csv_out, "--timeout", "1800"],
           cwd=ROOT, env=env, logf=logf, timeout=6 * 3600)
    paused_s = int(time.monotonic() - t0)
    rows = csv_rows(csv_out)
    journal = ROOT / "bench" / "bench-results.csv"  # the project's benchmark journal
    if rows and journal.exists():
        hdr = csv_header(journal)
        with open(journal, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
            w.writerows(rows)
    labels = list(dict.fromkeys(r["label"] for r in rows))
    dense = ",".join(lb for lb in labels if not lb.endswith("-sparse"))
    wins = [w.split("=")[0] for w in (P["windows"] + "," + P["big"]).split(",")]
    figs = []
    for win in wins:
        png = out_dir / f"waterfall-{win}-by-version.png"
        if sh([plot_python(), WATERFALL, csv_out, "--window", win, "--group-by", "label", "--label", dense,
               "--per", "1e10", "-o", png], logf=out_dir / "figures.log", timeout=300)[0] == 0:
            figs.append(png.name)
    png = out_dir / f"waterfall-{cur}-by-window.png"
    if sh([plot_python(), WATERFALL, csv_out, "--label", cur, "--group-by", "window", "--per", "1e10",
           "-o", png], logf=out_dir / "figures.log", timeout=300)[0] == 0:
        figs.append(png.name)
    md = [f"# Benchmark session {out_dir.name} ({profile})", "",
          f"Current release `{release_meta(act).get('id')}` ({cur}); arms: {', '.join(labels)}. Interleaved "
          f"fresh-process runs (tools/bench.py); production paused {paused_s} s. Rows appended to "
          "bench/bench-results.csv; figures from this session only.", "", "## Where the time goes",
          "", *bench_analysis(rows, cur, wins), "## Figures", "", *[f"![{f}]({f})" for f in figs]]
    (out_dir / "benchmark.md").write_text("\n".join(md) + "\n")
    log(f"benchmark ({profile}): {len(rows)} runs, {len(figs)} figures -> {out_dir / 'benchmark.md'}")
    return {"dir": str(out_dir), "profile": profile, "runs": len(rows), "figures": figs, "paused_s": paused_s}


BENCH_PRESETS = {"bench": 2e9, "steady": 2e10, "end": 2e10, "long": 1e11, "long-end": 1e11}


def window_span(spec):
    """Integers of a bench.py window spec (preset or NAME=START:LIMIT)."""
    if "=" in spec:
        a, b = spec.split("=", 1)[1].split(":")
        return int(b) - int(a)
    return BENCH_PRESETS.get(spec, 1e11)
