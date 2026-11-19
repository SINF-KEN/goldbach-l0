#!/usr/bin/env python3
"""figure_groups.py — sort the by-version figure labels by pipeline (stdlib only).

docs/make-plots.sh draws one figure per group, because the groups measure
different pipelines and are not comparable bar for bar:

  stats     dense path with exact statistics (v2.2.0+, the production path)
  prestats  dense path before the statistics (v1.x)
  sparse    the opt-in sparse path (GB_SPARSE=1, ~14x the dense GPU rate)

A label's pipeline is read from the rows docs/make-plots.sh will draw: per
window, the latest bench.py session (waterfall.py --latest). Rows written
since bench.py records the gb_env column say it themselves (GB_SPARSE=1);
older rows have no gb_env, and for them LEGACY_SPARSE lists the labels that
were benched only on the sparse path (HISTORY.md §9.2k). A label whose drawn
rows mix sparse and dense is an error: one of its sessions was benched
under the wrong label, and averaging or splitting it silently is how v2.0.0
and v2.1.0 ended up as the "fastest" versions in the dense figure.

    python3 ci/figure_groups.py bench/bench-results.csv v2.7.0,v2.6.0,...
    -> one line per non-empty group: "<group> <label>,<label>,..."
"""
import csv
import re
import sys
from collections import defaultdict

LEGACY_SPARSE = {"v2.0.0", "v2.1.0"}   # benched with GB_SPARSE=1 before gb_env existed
STATS_FROM = (2, 2, 0)                 # first version with exact statistics (dense)
WINDOWS = ("long", "steady")           # the windows of the by-version figures
GROUPS = ("stats", "prestats", "sparse")


def is_sparse(gb_env):
    env = dict(kv.split("=", 1) for kv in gb_env.split(",") if "=" in kv)
    return env.get("GB_SPARSE", "0") not in ("", "0")


def vkey(label):
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", label)
    return tuple(int(x) for x in m.groups()) if m else None


def classify(rows, labels):
    """rows: journal dicts; labels: ordered list. Returns {group: [labels]}."""
    latest = {}                         # (label, window) -> latest session
    for r in rows:
        if r.get("status") != "pass" or r.get("label") not in labels or r.get("window") not in WINDOWS:
            continue
        k = (r["label"], r["window"])
        latest[k] = max(latest.get(k, ""), r.get("session", ""))
    modes = defaultdict(set)            # label -> {"sparse", "dense"}
    where = defaultdict(list)
    for r in rows:
        k = (r.get("label"), r.get("window"))
        if k not in latest or r.get("status") != "pass" or r.get("session", "") != latest[k]:
            continue
        env = r.get("gb_env") or ""
        if env:
            mode = "sparse" if is_sparse(env) else "dense"
        else:
            mode = "sparse" if r["label"] in LEGACY_SPARSE else "dense"
        modes[r["label"]].add(mode)
        where[r["label"]].append(f"{r['window']} {r.get('session', '')} {env or '(no gb_env)'}")
    groups = {g: [] for g in GROUPS}
    errors = []
    for lb in labels:
        m = modes.get(lb, {"sparse" if lb in LEGACY_SPARSE else "dense"})
        if len(m) > 1:
            errors.append(f"{lb}: the rows drawn for it mix the sparse and the dense path:\n    "
                          + "\n    ".join(sorted(set(where[lb]))))
            continue
        if m == {"sparse"}:
            groups["sparse"].append(lb)
        elif (vkey(lb) or (0,)) >= STATS_FROM:
            groups["stats"].append(lb)
        else:
            groups["prestats"].append(lb)
    if errors:
        raise ValueError("\n".join(errors) + "\nre-bench these labels or give the sparse runs their own label")
    return groups


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1], newline="") as f:
        rows = list(csv.DictReader(f))
    labels = [s for s in sys.argv[2].split(",") if s]
    try:
        groups = classify(rows, labels)
    except ValueError as e:
        sys.exit(f"{sys.argv[1]}: {e}")
    for g in GROUPS:
        if groups[g]:
            print(g, ",".join(groups[g]))


if __name__ == "__main__":
    main()
