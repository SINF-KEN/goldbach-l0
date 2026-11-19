#!/usr/bin/env python3
"""version_bench.py — keep the benchmark journal and the README figures current (stdlib only).

Run by the GitHub workflow `Version figures` (.github/workflows/version-figures.yml)
whenever the VERSION in include/config.hpp changes on main, or by hand there
(workflow_dispatch). Two steps around ci/experiment.py, which does the
measuring (production paused, every arm a git ref built from source with its
committed kernels/goldbach_bmg.spv, interleaved fresh-process bench.py runs):

  spec    which versions to measure -> an experiment spec (timing only)
          auto: the VERSION at --ref and the one before it, unless the
          journal already holds the current one (then nothing to do)
  append  experiment bench.csv -> bench/bench-results.csv (commit and
          version columns set to the measured release) and every measured
          label into bench/figure-versions (the labels docs/make-plots.sh draws)

A version's release commit is the first commit on --ref's first-parent
history (main) whose include/config.hpp carries it. A plain "vX.Y.Z" label
therefore always means that code, not whatever was promoted later under the
same VERSION (the v2.7.0 speed rounds ran as "2.6.0" until the bump).
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_groups import is_sparse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = "include/config.hpp"
JOURNAL = ROOT / "bench" / "bench-results.csv"
FIGURE_VERSIONS = ROOT / "bench" / "figure-versions"
WINDOWS = "bench,steady,end,long,long-end"         # every window the three figures draw
VERSION_RE = re.compile(r'const char\* VERSION = "([0-9]+\.[0-9]+\.[0-9]+)"')
LABEL_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout


def version_at(rev):
    m = VERSION_RE.search(git("show", f"{rev}:{CONFIG}"))
    if not m:
        sys.exit(f"no VERSION in {CONFIG} at {rev}")
    return m.group(1)


def releases(ref):
    """[(version, first commit carrying it)] along ref's first-parent history, oldest first."""
    out, prev = [], None
    commits = git("log", "--first-parent", "--reverse", "--format=%H", ref, "--", CONFIG).split()
    for c in commits:
        v = version_at(c)
        if v != prev:
            out.append((v, c))
            prev = v
    return out


def vkey(label):
    return tuple(int(x) for x in label.lstrip("v").split("."))


def journal_labels(journal, window="long"):
    if not journal.exists():
        return set()
    with open(journal, newline="") as f:
        return {r["label"] for r in csv.DictReader(f) if r.get("window") == window and r.get("status") == "pass"}


def cmd_spec(a):
    rel = releases(a.ref)
    first = {}
    for v, c in rel:
        first.setdefault(v, c)                     # a version that came back later keeps its first commit
    if a.versions == "auto":
        cur = version_at(a.ref)
        order = [v for v, _ in rel]
        i = max(k for k, v in enumerate(order) if v == cur)
        want = [cur] + ([order[i - 1]] if i > 0 else [])
        if f"v{cur}" in journal_labels(a.journal) and not a.force:
            print(f"v{cur} already in {a.journal}: nothing to measure (--force to repeat)")
            want = []
    else:
        want = [v.strip().lstrip("v") for v in a.versions.split(",") if v.strip()]
    arms, pins = [], dict(p.split("=", 1) for p in a.pin)
    for v in want:
        c = pins.get(f"v{v}") or first.get(v)
        if not c:
            sys.exit(f"v{v}: not a VERSION on the first-parent history of {a.ref} "
                     f"(known: {', '.join(sorted({x for x, _ in rel}, key=lambda s: vkey(s)))})")
        c = git("rev-parse", c).strip()
        arms.append({"label": f"v{v}", "ref": c, "spv": "shipped", "timing_only": True})
        print(f"v{v}: release commit {c[:12]}")
    spec = {"note": "Version figures: " + ", ".join(f"`{x['label']}` = {x['ref'][:12]}" for x in arms)
                    + " (first commit with that VERSION on main), shipped kernels, production configuration.",
            "windows": a.windows, "runs": a.runs, "warmup": 1, "exact_window": "",
            "ref_arm": arms[-1]["label"] if arms else "", "arms": arms}
    a.out.write_text(json.dumps(spec, indent=2) + "\n")
    if a.github_output:
        with open(a.github_output, "a") as f:
            f.write(f"todo={'1' if arms else '0'}\n")
            f.write(f"labels={','.join(x['label'] for x in arms)}\n")
            f.write(f"current={arms[0]['label'] if arms else ''}\n")


def cmd_append(a):
    spec = json.loads(a.spec.read_text())
    commit = {x["label"]: x["ref"][:7] for x in spec["arms"]}
    with open(a.bench_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label") in commit]
    if not any(r.get("status") == "pass" for r in rows):
        sys.exit(f"{a.bench_csv}: no passing rows for {', '.join(commit)}")
    # the figures draw a version's dense production path; a sparse run under a
    # version label is how v2.0.0/v2.1.0 once looked ~14x faster than the rest
    sparse = sorted({r["label"] for r in rows if is_sparse(r.get("gb_env") or "")})
    if sparse:
        sys.exit(f"{a.bench_csv}: GB_SPARSE rows for {', '.join(sparse)}; the version figures "
                 "measure the dense production path (give sparse runs their own label)")
    for r in rows:                                 # bench.py writes the checkout it ran in (production's tree)
        r["commit"] = commit[r["label"]]
        r["version"] = r["label"][1:]
    with open(a.journal, newline="") as f:
        old = list(csv.reader(f))
    hdr = old[0]
    with open(a.bench_csv, newline="") as f:
        new_cols = [c for c in (next(csv.reader(f), None) or []) if c not in hdr]
    if new_cols:                                   # columns bench.py added since (gb_env): as its upgrade_journal
        hdr += new_cols
        with open(a.journal, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(r + [""] * len(new_cols) for r in old[1:])
        print(f"journal: added column(s) {', '.join(new_cols)}")
    with open(a.journal, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore").writerows(rows)
    print(f"{len(rows)} rows -> {a.journal}")

    lines = a.figure_versions.read_text().splitlines() if a.figure_versions.exists() else []
    head = [s for s in lines if s.strip().startswith("#") or not s.strip()]
    have = [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]
    new = [lb for lb in commit if lb not in have]
    if new:
        labels = sorted(set(have) | set(new), key=lambda s: vkey(s) if LABEL_RE.match(s) else (0,), reverse=True)
        a.figure_versions.write_text("\n".join(head + labels) + "\n")
        print(f"{', '.join(new)} -> {a.figure_versions}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec", help="versions to measure -> experiment spec")
    s.add_argument("--versions", default="auto", help="auto, or e.g. v2.7.0,v2.6.0 (first = current)")
    s.add_argument("--pin", action="append", default=[], metavar="vX.Y.Z=REV",
                   help="measure this commit for a version instead of its first commit on main")
    s.add_argument("--ref", default="HEAD")
    s.add_argument("--force", action="store_true", help="auto: measure even if the journal has the version")
    s.add_argument("--windows", default=WINDOWS)
    s.add_argument("--runs", type=int, default=5)
    s.add_argument("--journal", type=Path, default=JOURNAL)
    s.add_argument("-o", "--out", type=Path, required=True)
    s.add_argument("--github-output", type=Path)
    s.set_defaults(fn=cmd_spec)
    p = sub.add_parser("append", help="experiment bench.csv -> journal + figure-versions")
    p.add_argument("bench_csv", type=Path)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--journal", type=Path, default=JOURNAL)
    p.add_argument("--figure-versions", type=Path, default=FIGURE_VERSIONS)
    p.set_defaults(fn=cmd_append)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
