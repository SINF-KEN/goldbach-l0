#!/usr/bin/env python3
"""test_figure_groups.py — ci/figure_groups.py sorts figure labels by pipeline.

    python3 tests/test_figure_groups.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ci"))
import figure_groups as fg  # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    fails += not ok


def row(label, window="long", session="2026-11-19T10:00:00+00:00", gb_env=None, status="pass"):
    r = {"label": label, "window": window, "session": session, "status": status}
    if gb_env is not None:
        r["gb_env"] = gb_env
    return r


def raises(rows, labels):
    try:
        fg.classify(rows, labels)
    except ValueError:
        return True
    return False


L = ["v2.7.0", "v2.3.0", "v2.1.0", "v2.0.0", "v1.13.0"]
old = [row(lb) for lb in L]                       # journal rows before gb_env
g = fg.classify(old, L)
check(g == {"stats": ["v2.7.0", "v2.3.0"], "prestats": ["v1.13.0"], "sparse": ["v2.1.0", "v2.0.0"]},
      "rows without gb_env: legacy sparse list, stats from v2.2.0")

g = fg.classify([row("v2.8.0", gb_env="GB_BUCKET_VERIFY=100,GB_SPARSE=1")], ["v2.8.0"])
check(g["sparse"] == ["v2.8.0"], "gb_env GB_SPARSE=1 -> sparse, whatever the version")

g = fg.classify([row("v2.8.0", gb_env="GB_BUCKET_VERIFY=100,GB_SPARSE=0")], ["v2.8.0"])
check(g["stats"] == ["v2.8.0"], "GB_SPARSE=0 -> dense")

g = fg.classify([row("v2.1.0", gb_env="GB_BUCKET_VERIFY=100")], ["v2.1.0"])
check(g["stats"] == [] and g["prestats"] == ["v2.1.0"], "a legacy label re-benched dense (gb_env set) is dense")

check(raises([row("v2.8.0", gb_env="GB_BUCKET_VERIFY=100"),
              row("v2.8.0", window="steady", gb_env="GB_SPARSE=1")], ["v2.8.0"]),
      "sparse in one window, dense in another -> error")
check(raises([row("v2.8.0", gb_env="GB_BUCKET_VERIFY=100"),
              row("v2.8.0", gb_env="GB_SPARSE=1")], ["v2.8.0"]),
      "sparse and dense rows in one drawn session -> error")

g = fg.classify([row("v2.8.0", session="2026-09-01T00:00:00+00:00", gb_env="GB_SPARSE=1"),
                 row("v2.8.0", gb_env="GB_BUCKET_VERIFY=100")], ["v2.8.0"])
check(g["stats"] == ["v2.8.0"], "only the latest session counts (waterfall.py --latest)")

g = fg.classify([row("v2.8.0", gb_env="GB_SPARSE=1", status="fail"),
                 row("v2.8.0", gb_env="")], ["v2.8.0"])
check(g["stats"] == ["v2.8.0"], "failed rows are ignored")

g = fg.classify([row("v2.8.0", window="bench", gb_env="GB_SPARSE=1"), row("v2.8.0")], ["v2.8.0"])
check(g["stats"] == ["v2.8.0"], "windows the by-version figures do not draw are ignored")

g = fg.classify([], ["v1.6.2"])
check(g["prestats"] == ["v1.6.2"], "a label without rows still lands in its version group")

print("FAILED" if fails else "all passed")
sys.exit(1 if fails else 0)
